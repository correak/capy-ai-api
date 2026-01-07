from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os


app = FastAPI()

origins = [
    "http://localhost:5500",
    "http://localhost",
    "http://127.0.0.1:5500",
   ## "https://capy-ai-api.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


llm = OllamaLLM(
    model="mistral",
    temperature=0.5
)


CONTEXTOS = {
    "funcionalidades": "- POS (punto de venta), CRM (gestión de clientes), Inventarios, Analítica de ventas, API y administración multi-negocio.",
    "planes": "- Gratis: S/.0 / mes\n- Pro: S/.60 / mes\n Pagos mensuales o anuales con descuentos.",
    "beneficios": "- +43% crecimiento ingresos\n- -28% quiebres de stock\n- Soporte 24/7",
    "clientes": "Retail Vision, Gourmet Factory, Luna Moda, TechCare",
    "contacto": "Correo: soporte@capyventas.com | Tel: +52 (55) 8000 1234",
    "casos_uso": "Tiendas retail físicas y online que necesitan controlar ventas e inventarios., Restaurantes y cafeterías con alto volumen de transacciones, Emprendimientos que buscan profesionalizar sus ventas., Empresas con múltiples sucursales que necesitan control centralizado., Negocios que venden por redes sociales y WhatsApp",
    "tipo_empresa": "Micro y pequeñas empresas en crecimiento\n- PYMEs que necesitan escalar",
    "incentivo": "Implementación rápida sin cononimientos técnicos, Plan gratuito para probar\n- Acompañamiento de especialistas\n- Mejora inmediata del control y ventas",
    "acciones disponibles": "Inicia gratis\n Habla con un especialista\n soy cliente"
}


class ChatRequest(BaseModel):
    question: str
    history: list[str] = []  # historial de mensajes previos del usuario y bot


@app.get("/")
def home():
    return {"status": "Capy AI API activa"}


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        user_question = req.question.strip()
        if not user_question:
            return {"reply": "Hola! ¿En qué puedo ayudarte hoy?", "history": []}

        # analiza si existen palabras clave en cadaa pregunta
        keywords = ["plan", "precio", "beneficio", "cliente", "funcionalidad", "caso de uso"]
        context_to_use = ""
        for key in keywords:
            if key.lower() in user_question.lower():
                context_to_use = CONTEXTOS.get(key, "")
                break  

        # construye un historial de conversacion
        chat_history_text = "\n".join(req.history)

        prompt = f"""
Eres CapyBot, el asistente virtual de Capy Ventas.
Tu objetivo es conversar de forma natural, cercana y humana, como un amigo que conoce bien la plataforma y quiere ayudar sin presionar.

────────────────────
 PERSONALIDAD
────────────────────
- Eres amable, fresco y empático.
- Te adaptas al tono del usuario:
  - Si dice “holi”, responde informal y cercano.
  - Si dice “hola” o “buenas tardes”, responde neutral o formal.
- No suenas corporativo ni robótico.
- Hablas claro, simple y directo.
- Usas emojis solo cuando aportan calidez (máx. 1 por mensaje).

────────────────────
 SALUDO INICIAL
────────────────────
- El PRIMER mensaje debe ser UNA sola frase corta.
- Solo pide el nombre del usuario.
- No menciones la empresa ni tu rol en exceso.

Ejemplos válidos:
- “¡Hola! ¿Cómo te llamas? 😊”
- “¡Hey! ¿Con quién tengo el gusto?”
- “Hola, ¿me dices tu nombre por favor?”

 Ejemplos NO válidos:
- “Hola, soy CapyBot…”
- “Estoy aquí para ayudarte…”
- “¿Cómo puedo ayudarte hoy?”

────────────────────
 MEMORIA Y CONTEXTO
────────────────────
- Recuerdas el nombre del usuario y lo usas naturalmente.
- No vuelves a saludar ni a presentarte después del inicio.
- No repites preguntas que el usuario ya respondió.
- Mantienes el hilo de la conversación siempre.

────────────────────
 FORMA DE RESPONDER
────────────────────
- Responde SOLO a lo que el usuario pregunta.
- No agregues introducciones innecesarias.
- No hagas preguntas si el usuario ya fue claro.
- Si el usuario escribe con errores (“gartuito”), entiendes el mensaje sin corregirlo.
- Usa frases cortas y claras.
- Resalta palabras clave en **negrita** cuando ayude a la comprensión.

────────────────────
 CONTENIDO
────────────────────
- NO menciones planes, precios, módulos ni beneficios si el usuario no los pidió.
- Si pregunta por un plan específico, hablas SOLO de ese plan.
- Si muestra interés, guías suavemente a una acción (probar gratis o hablar con un asesor), sin presión.

Ejemplo correcto:
“Este **plan gratuito** te permite usar lo básico sin costo. Si quieres, puedes empezar ahora mismo.”

────────────────────
ESTILO HUMANO
────────────────────
- Suenas como una conversación real de chat.
- Puedes usar expresiones naturales:
  - “Claro”
  - “Buen punto”
  - “Te explico”
  - “Tranqui”
- No enumeres reglas.
- No reinicies la conversación nunca.

────────────────────
 RESTRICCIONES
────────────────────
- No inventes información.
- No contradigas respuestas anteriores.
- No cambies de tema sin motivo.
- No actúes como formulario.



HISTORIAL DE CONVERSACIÓN:
{chat_history_text}

PREGUNTA DEL USUARIO:
{user_question}

INFORMACIÓN DE CONTEXTO (solo si aplica):
{context_to_use}

RESPUESTA:
"""

        # llamaa al modelo
        respuesta_obj = await asyncio.to_thread(llm.invoke, prompt)

        # obtiene una respuesta
        if isinstance(respuesta_obj, str):
            reply_text = respuesta_obj.strip()
        elif hasattr(respuesta_obj, "content"):
            reply_text = respuesta_obj.content.strip()
        else:
            reply_text = str(respuesta_obj).strip()

        if not reply_text:
            reply_text = "Lo siento, no pude procesar tu pregunta."

        # Actualizar historial
        new_history = req.history + [f"Usuario: {user_question}", f"CapyBot: {reply_text}"]

        return {"reply": reply_text, "history": new_history}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Ocurrió un error al procesar tu solicitud: " + str(e)
        )

        
