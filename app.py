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
            return {"reply": "¿En qué puedo ayudarte?", "history": req.history}

        # 
        nombre_usuario = extraer_nombre(req.history)

        # 
        context_to_use = ""
        for key, value in CONTEXTOS.items():
            if key in user_question.lower():
                context_to_use += value + " "

        # 
        reglas = f"""
        Eres CapyBot, asistente inteligente de Capy Ventas.
        Usuario: {nombre_usuario if nombre_usuario else 'Desconocido'}

        REGLAS CRÍTICAS:
        1. Responde SIEMPRE en el MISMO IDIOMA en el que el usuario te escriba.
        2. Si el usuario te pide cambiar de idioma, hazlo inmediatamente.
        3. SOLO responde sobre Capy Ventas (POS, inventarios, planes, registro).
        4. Si preguntan sobre otros temas (Biblia, plantas, etc.), responde amablemente en el idioma del usuario que solo eres un asistente de Capy Ventas.
        5. Si el contexto está en español y el usuario habla otro idioma, TRADUCE la información.
        6. Si no conoces el nombre del usuario, pregúntaselo amablemente.
        7. 4. Usa emojis apropiadamente para hacer la conversación más amigable.
        8. Inicia la conversación pidiendole su nombre si no lo sabes y no ha saludado antes.
        7. Registro: http://localhost/capy-ventas/pos/login
        """

        # 4. Prompt limpio
        chat_history_text = "\n".join(req.history[-10:])
        prompt = f"""{reglas}
        
        Contexto: {context_to_use}
        Historial: {chat_history_text}
        Pregunta: {user_question}
        Respuesta:"""

        # 5. Ejecución
        response = await asyncio.to_thread(llm.invoke, prompt)
        reply_text = response.content.strip()

        new_history = req.history + [f"Usuario: {user_question}", f"CapyBot: {reply_text}"]

        return {
            "reply": reply_text,
            "history": new_history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))