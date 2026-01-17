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
            return {"reply": "¡Hola! 😊 ¿Cómo te llamas?", "history": req.history}

        nombre_usuario = extraer_nombre(req.history)

        # looking for context keywords
        context_to_use = ""
        for key, value in CONTEXTOS.items():
            if key in user_question.lower():
                context_to_use += value + " "

        # role and rules for the assistant
        reglas = f"""
        You are CapyBot, a friendly assistant for Capy Ventas.
        User Name: {nombre_usuario if nombre_usuario else 'Unknown'}

        STRICT RULES:
        1. LANGUAGE: Always detect the user's language and respond in that EXACT same language. 
        2. TRANSLATION: If the provided context is in Spanish, translate it accurately to the user's language.
        3. SCOPE: Only answer questions about Capy Ventas (POS, inventory, plans, benefits). 
        4. OFF-TOPIC: If the user asks about unrelated topics (Religion, plants, etc.), politely decline in their language.
        5. NAME: If the user's name is 'Unknown', you MUST ask for their name in your response.
        6. STYLE: Be very brief, friendly, and use emojis.
        7. CONTEXT: Use the following information to answer: {context_to_use}
        8. URL: http://localhost/capy-ventas/pos/login
        """

        chat_history_text = "\n".join(req.history[-10:])
        
        # estructuring the prompt
        prompt = f"""
        SYSTEM: {reglas}
        CHAT HISTORY: {chat_history_text}
        USER MESSAGE: {user_question}
        ASSISTANT RESPONSE:"""

        response = await asyncio.to_thread(llm.invoke, prompt)
        reply_text = response.content.strip()

        new_history = req.history + [f"Usuario: {user_question}", f"CapyBot: {reply_text}"]

        return {
            "reply": reply_text,
            "history": new_history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))