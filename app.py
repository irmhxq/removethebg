import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from transparent_background import Remover

# Verificación de hardware
print(f"¿GPU AMD (ROCm) detectada y lista?: {torch.cuda.is_available()}")
# Nota: PyTorch ROCm abstrae la GPU AMD y la etiqueta internamente como "cuda"

print("Cargando BiRefNet...")
# BiRefNet de ZhengPeng7 requiere trust_remote_code=True
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.eval().to("cuda")

print("Cargando InSPyReNet...")
# Remover encapsula InSPyReNet y usa la GPU (cuda) de forma predeterminada si está disponible
inspyrenet = Remover(mode="base")
print("Modelos cargados exitosamente. Lanzando UI...")


def remove_bg_birefnet(image):
    # Asegurar base RGBA para manipular la transparencia final
    image_rgba = image.convert("RGBA")
    image_rgb = image.convert("RGB")

    # Pre-procesamiento requerido por BiRefNet
    transform_image = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = transform_image(image_rgb).unsqueeze(0).to("cuda")

    # Inferencia
    with torch.no_grad():
        preds = birefnet(input_tensor)[-1].sigmoid().cpu()

    pred = preds[0].squeeze()

    # Convertir el mapa de predicciones a una máscara (Alpha) y redimensionar al original
    mask = transforms.ToPILImage()(pred).resize(image.size, Image.Resampling.BILINEAR)
    image_rgba.putalpha(mask)

    return image_rgba


def remove_bg_inspyrenet(image):
    # transparent-background nativamente procesa PIL images a RGBA transparentes
    return inspyrenet.process(image)


def process_image(image, model_choice):
    if image is None:
        return None
    if model_choice == "BiRefNet":
        return remove_bg_birefnet(image)
    else:
        return remove_bg_inspyrenet(image)


# --- Interfaz de Usuario con Gradio ---
with gr.Blocks(
    title="Removedor de Fondos (AMD GPU)", theme=gr.themes.Monochrome()
) as app:
    gr.Markdown(
        """
        # ✂️ Removedor de Fondos Minimalista
        Aplicación acelerada localmente en hardware **AMD (Linux ROCm)**. 
        Elige entre **BiRefNet** (alta precisión en bordes finos) e **InSPyReNet** (excelente balance general).
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Sube tu Imagen")
            model_selector = gr.Radio(
                choices=["BiRefNet", "InSPyReNet"],
                value="BiRefNet",
                label="Selecciona el Modelo IA",
            )
            submit_btn = gr.Button("Remover Fondo", variant="primary")

        with gr.Column():
            # format="png" y image_mode="RGBA" aseguran que el output preserve el canal de transparencia
            output_image = gr.Image(
                type="pil",
                image_mode="RGBA",
                label="Resultado (PNG Transparente)",
                format="png",
            )

    submit_btn.click(
        fn=process_image, inputs=[input_image, model_selector], outputs=output_image
    )

if __name__ == "__main__":
    # Si usas el PC localmente:
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)

    # SI ESTÁS EN UN SERVIDOR REMOTO O DOCKER, usa esta en su lugar:
    # app.launch(server_name="0.0.0.0", server_port=7860, share=False)
