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
            return {"reply": "¿Cómo te llamas?", "history": req.history}

        # detectar el idioma
        try:
            lang = detect(user_question)
        except:
            lang = "es"
        
        idioma = "en" if lang == "en" else "es"

        # Verificar si ya tenemos el nombre
        nombre_usuario = extraer_nombre(req.history)

        # buqueda de de contexto relevante
        context_to_use = ""
        for key, value in CONTEXTOS.items():
            if key in user_question.lower():
                context_to_use += value + " "

        # construcción de Reglas Dinámicas

        if idioma == "es":
            instruccion_nombre = (
                "Si no conoces el nombre del usuario, preséntate como CapyBot y pregúntale su nombre." 
                if not nombre_usuario else f"Saluda a {nombre_usuario}."
            )
            reglas = f"""
            Eres CapyBot, un asistente EXCLUSIVO de Capy Ventas.
            {instruccion_nombre}

            REGLAS DE ORO:
            1. SOLO puedes responder sobre Capy Ventas (POS, inventarios, planes, precios, registro).
            2. Si el usuario pregunta sobre CUALQUIER otro tema (la Biblia, plantas, cocina, otros softwares, etc.), debes responder exactamente: "Lo siento, como asistente de Capy Ventas, solo puedo ayudarte con temas relacionados a nuestra plataforma. ¿Tienes alguna duda sobre nuestros planes o el POS?"
            3. Responde siempre en ESPAÑOL.
            4. Usa emojis y sé muy breve.
            5. Registro: http://localhost/capy-ventas/pos/login
            """
        else:
            instruccion_nombre = (
                "If you don't know the user's name, ask for it politely." 
                if not nombre_usuario else f"Greet {nombre_usuario}."
            )
            reglas = f"""
            You are CapyBot, an EXCLUSIVE assistant for Capy Ventas.
            {instruccion_nombre}

            GOLDEN RULES:
            1. ONLY answer questions about Capy Ventas (POS, inventory, plans, pricing, registration).
            2. If the user asks about ANY other topic (the Bible, plants, cooking, other software, etc.), you must reply: "I'm sorry, as a Capy Ventas assistant, I can only help you with topics related to our platform. Do you have any questions about our plans or the POS?"
            3. ALWAYS respond in ENGLISH.
            4. Use emojis and be very concise.
            5. Registration: http://localhost/capy-ventas/pos/login
            """
        # preparar el Prompt para el LLM
        chat_history_text = "\n".join(req.history[-10:])
        prompt = f"""
        {reglas}

        Context information:
        {context_to_use}

        Chat History:
        {chat_history_text}

        User Question: {user_question}
        
        Assistant Response:"""

        # eejecución
        response = await asyncio.to_thread(llm.invoke, prompt)
        reply_text = response.content.strip()

        # actualizar Historial
        new_history = req.history + [f"Usuario: {user_question}", f"CapyBot: {reply_text}"]

        return {
            "reply": reply_text,
            "history": new_history,
            "language_detected": idioma
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))