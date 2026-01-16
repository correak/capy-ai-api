from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langdetect import detect 
import os
import asyncio


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature = 0.5,
    api_key=os.getenv("GROQ_API_KEY")
)


CONTEXTOS = {
    "plan": "Planes disponibles:\n- **Gratis**: funciones básicas para empezar\n- **Pro**: herramientas avanzadas para crecer",
    "precio": "El plan **Pro** cuesta S/. 60 al mes. El plan **Gratis** no tiene costo.",
    "beneficio": "Mejora el control de ventas, reduce errores y ahorra tiempo.",
    "cliente": "Empresas de retail, restaurantes y emprendimientos en crecimiento.",
    "funcionalidad": "POS, CRM, inventarios, reportes y control multi-sucursal.",
    "caso de uso": "Ideal para negocios que venden en tienda física, online o por WhatsApp."
}

class ChatRequest(BaseModel):
    question: str
    history: list[str] = []

def extraer_nombre(history: list[str]):
    for h in history:
        if h.startswith("Usuario:"):
            posible = h.replace("Usuario:", "").strip()
            if 1 <= len(posible.split()) <= 2:
                return posible
    return None

def saludo_ya_realizado(history: list[str]):
    return any("Hola" in h or "Hi" in h for h in history)

@app.get("/")
def home():
    return {"message": "Backend's ready to use"}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        user_question = req.question.strip()
        if not user_question:
            return {"reply": "¿Me repites eso porfa?", "history": req.history}

        nombre_usuario = extraer_nombre(req.history)
        saludo_hecho = saludo_ya_realizado(req.history)

        # 
        idioma = "en" if detect(user_question) == "en" else "es"

        # 
        context_to_use = ""
        for key in CONTEXTOS:
            if key in user_question.lower():
                context_to_use = CONTEXTOS[key]
                break

        # 
        chat_history_text = "\n".join(req.history[-20:])

        # 
        if idioma == "es":
            reglas = f"""
Eres CapyBot, asistente cercano y humano de Capy Ventas.

Reglas:
- Solo responde preguntas relacionadas con Capy Ventas.
- Saluda automáticamente si es la primera interacción.
- Usa el nombre del usuario si lo conoces: {nombre_usuario if nombre_usuario else 'Desconocido'}
- Responde directo, claro y conciso.
- Usa emojis y frases cercanas.
- Usa listas numeradas solo si la pregunta lo requiere.
- Si el usuario pide cambiar de idioma, hazlo inmediatamente.
- No hables de planes o precios si no se preguntó.
- Incita suavemente a registrarse: http://localhost/capy-ventas/pos/login
- Ignora preguntas fuera de contexto y di: "No tengo información sobre eso".
"""
        else:  # inglés
            reglas = f"""
You are CapyBot, a friendly assistant for Capy Ventas.

Rules:
- Only answer questions about Capy Ventas.
- Automatically greet the user if this is the first interaction.
- Use the user's name if known: {nombre_usuario if nombre_usuario else 'Unknown'}
- Answer clearly and concisely.
- Use emojis and friendly expressions.
- Use numbered lists only if the question requires it.
- Do not talk about plans or prices unless asked.
- Gently encourage registration: http://localhost/capy-ventas/pos/login
- Ignore unrelated questions and say: "I don't have information about that."
"""

        prompt = f"""
{reglas}

History:
{chat_history_text}

Question:
{user_question}

Context:
{context_to_use}

Response:
"""

        # 
        respuesta_obj = await asyncio.to_thread(llm.invoke, prompt)

        if hasattr(respuesta_obj, "content"):
            reply_text = respuesta_obj.content.strip()
        else:
            reply_text = str(respuesta_obj).strip()

        #
        if not reply_text:
            reply_text = "No te entendí bien, ¿me explicas un poquito más?" if idioma == "es" else "I didn't quite understand, could you explain a bit more?"

        # 
        new_history = req.history + [
            f"Usuario: {user_question}",
            f"CapyBot: {reply_text}"
        ]

        return {
            "reply": reply_text,
            "history": new_history
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ocurrió un error: {str(e)}"
        )
