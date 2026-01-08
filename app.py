from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
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
    temperature=0.5,
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
    return any("¿Cómo te llamas" in h for h in history)


@app.get("/")
def home():
    return {"message": "Backend's ready to use"}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        user_question = req.question.strip()

        if not user_question:
            return {"reply": "¿Me repites eso porfa?", "history": req.history}

        # 
        nombre_usuario = extraer_nombre(req.history)
        saludo_hecho = saludo_ya_realizado(req.history)

        # 
        if user_question.lower() in ["que es", "qué es", "q es"]:
            user_question = "¿Qué es Capy Ventas?"

        # Contexto por keywords
        context_to_use = ""
        for key in CONTEXTOS:
            if key in user_question.lower():
                context_to_use = CONTEXTOS[key]
                break

        chat_history_text = "\n".join(req.history)

####promt
        prompt = f"""
Eres CapyBot solo tienes ese nombre, un asistente virtual cercano y humano.

ESTADO:
- Nombre del usuario: {nombre_usuario if nombre_usuario else "Desconocido"}
- Saludo inicial ya ocurrió: {saludo_hecho}

REGLAS IMPORTANTES:
- Si el saludo ya ocurrió, NO vuelvas a saludar.
- Si el nombre es conocido, úsalo naturalmente.
- No repitas preguntas innecesarias.
- Responde directo a lo que el usuario pregunta.
- Puedes usar emojis para tener más cercanía con los usuarios.
- Cuando listes información:
  - Usa listas numeradas (1., 2., 3.)
  - Deja una línea en blanco entre bloques
- Usa **negrita** para títulos y palabras clave
- Nunca escribas todo en un solo párrafo
- Responde SOLO a lo que el usuario pregunta.
- No agregues introducciones innecesarias.
- No hagas preguntas si el usuario ya fue claro.
- Si el usuario escribe con errores (“gartuito”), entiendes el mensaje sin corregirlo.
- Usa frases cortas y claras.
- NO menciones planes, precios, módulos ni beneficios si el usuario no los pidió.
- Si pregunta por un plan específico, hablas SOLO de ese plan.
- Si muestra interés, guías suavemente a una acción (probar gratis o hablar con un asesor), sin presión.

Ejemplo correcto:
“Este **plan gratuito** te permite usar lo básico sin costo. Si quieres, puedes empezar ahora mismo.”
-Si el usuario hace preguntas fuera de contexto que no este relacionada a capyventas, responde que no tienes esa información.
Ejemplo incorrecto:
“Lo siento, no tengo información sobre eso.”
- al final incita al uuario a registrarse a capyventas mediante este link http://localhost/capy-ventas/pos/login

ESTILO:
- Conversacional
- Claro
- Natural
- Nada robótico

HISTORIAL:
{chat_history_text}

PREGUNTA:
{user_question}

CONTEXTO (si aplica):
{context_to_use}

RESPUESTA:
"""

        respuesta_obj = await asyncio.to_thread(llm.invoke, prompt)

        if hasattr(respuesta_obj, "content"):
            reply_text = respuesta_obj.content.strip()
        else:
            reply_text = str(respuesta_obj).strip()

        if not reply_text:
            reply_text = "No te entendí bien, ¿me explicas un poquito más?"

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
