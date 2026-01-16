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
        
        # detectar el idioma más flexible
        try:
            idioma_detectado = detect(user_question)
        except:
            idioma_detectado = "es"
        
        idioma = "en" if idioma_detectado == "en" else "es"

        # busqueda del contexto
        context_to_use = ""
        for key in CONTEXTOS:
            if key in user_question.lower():
                context_to_use = CONTEXTOS[key]
                break

        chat_history_text = "\n".join(req.history[-10:])

        # aca el bot inicia el chat preguntando quien es el usuario y como se llama
        if idioma == "es":
            instruccion_nombre = "Si no conoces el nombre del usuario, pregúntale amablemente cómo se llama." if not nombre_usuario else f"Saluda a {nombre_usuario}."
            reglas = f"""
            Eres CapyBot, un asistente amigable de Capy Ventas.
            REGLA CRÍTICA: Responde SIEMPRE en ESPAÑOL.
            {instruccion_nombre}
            - Solo habla de Capy Ventas.
            - Si el contexto tiene información, úsala. Si no, usa tu conocimiento general sobre POS/Ventas.
            - Registro: http://localhost/capy-ventas/pos/login
            """
        else:
            instruccion_nombre = "If you don't know the user's name, ask for it politely." if not nombre_usuario else f"Greet {nombre_usuario}."
            reglas = f"""
            You are CapyBot, a friendly assistant for Capy Ventas.
            CRITICAL RULE: Always respond in ENGLISH.
            {instruccion_nombre}
            - Only talk about Capy Ventas.
            - If context is provided in another language, TRANSLATE it to English for the user.
            - Registration: http://localhost/capy-ventas/pos/login
            """

        prompt = f"{reglas}\n\nHistory:\n{chat_history_text}\n\nContext:\n{context_to_use}\n\nQuestion: {user_question}\n\nResponse:"

        respuesta_obj = await asyncio.to_thread(llm.invoke, prompt)
        reply_text = respuesta_obj.content.strip()

        new_history = req.history + [f"Usuario: {user_question}", f"CapyBot: {reply_text}"]
        return {"reply": reply_text, "history": new_history}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))