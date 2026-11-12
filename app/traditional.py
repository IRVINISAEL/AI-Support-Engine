import unicodedata
from app.schemas import AnalyzeResponse


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


KEYWORDS = {
    "SECURITY": [
        "vulnerab", "hackeo", "hackea", "inyeccion", "filtracion", "brecha",
        "phishing", "malware", "ransomware", "virus", "robo de", "xss",
        "acceso no autorizado", "exploit", "fuga de datos", "datos expuestos",
        "expuest", "suplant", "fraude", "sospechos",
    ],
    "ACCESS": [
        "iniciar sesion", "inicio de sesion", "login", "log in", "contrasena",
        "password", "clave", "no puedo entrar", "no puedo acceder", "acceso",
        "bloquead", "credenciales", "2fa", "verificacion en dos pasos",
    ],
    "BUG": [
        "error", "falla", "fallo", "bug", "se cierra", "se congela", "se traba",
        "no funciona", "no carga", "caido", "caida", "pantalla en blanco",
        "crash", "lento", "excepcion", "no responde", "se cae", "no sirve",
        "no abre",
    ],
    "FEATURE": [
        "me gustaria", "agrega", "anadir", "anade", "nueva funcion",
        "funcionalidad", "sugerencia", "seria util", "podrian", "quisiera",
        "modo oscuro", "mejorar", "implementar", "soporte para", "opcion de",
    ],
}

SECURITY_CRITICAL = [
    "filtracion", "hackeo", "hackea", "robo de", "ransomware", "brecha",
    "inyeccion", "datos expuestos", "fuga de datos", "acceso no autorizado",
    "expuest",
]
BUG_HIGH = [
    "caido", "caida", "ningun usuario", "todos los usuarios", "produccion",
    "perdi", "perdida de datos", "urgente", "no funciona nada",
]
ACCESS_HIGH = ["urgente", "bloquead"]

RESPONSES = {
    "ACCESS": "Lamentamos el inconveniente con su acceso. Verifique sus credenciales y utilice la opción de recuperación de contraseña si es necesario. Si el problema continúa, nuestro equipo le ayudará.",
    "BUG": "Gracias por reportar la falla. Hemos registrado el problema y nuestro equipo técnico lo revisará. Si puede, envíenos los pasos para reproducirlo.",
    "FEATURE": "Gracias por su sugerencia. La hemos registrado y será evaluada por nuestro equipo de producto para futuras versiones.",
    "SECURITY": "Gracias por informarnos. Su reporte se ha escalado de inmediato a nuestro equipo de seguridad para revisión humana. Por favor no comparta detalles sensibles por este medio.",
}


def _has(text: str, words: list[str]) -> bool:
    return any(w in text for w in words)


def is_security_related(message: str) -> bool:
    return _has(normalize(message), KEYWORDS["SECURITY"])


def analyze_traditional(message: str) -> AnalyzeResponse:
    text = normalize(message)
    scores = {cat: sum(1 for w in words if w in text) for cat, words in KEYWORDS.items()}

    low_confidence = False
    if scores["SECURITY"] > 0:
        category = "SECURITY"
    else:
        best = max(scores.values())
        if best == 0:
            category, low_confidence = "BUG", True
        else:
            category = next(c for c in ["ACCESS", "BUG", "FEATURE"] if scores[c] == best)

    if category == "SECURITY":
        priority = "CRITICAL" if _has(text, SECURITY_CRITICAL) else "HIGH"
    elif category == "BUG":
        priority = "LOW" if low_confidence else ("HIGH" if _has(text, BUG_HIGH) else "MEDIUM")
    elif category == "ACCESS":
        priority = "HIGH" if _has(text, ACCESS_HIGH) else "MEDIUM"
    else:
        priority = "LOW"

    return AnalyzeResponse(
        category=category,
        priority=priority,
        response=RESPONSES[category],
        requiresHumanReview=(category == "SECURITY" or low_confidence),
        mode="traditional",
    )