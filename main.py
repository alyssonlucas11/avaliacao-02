# =============================================================================
# ETAPA 4 — API RESTful com 4 Modelos LoRA (2 Causais + 2 Seq2Seq)
# Disciplina: Tópicos Avançados em IA A — CERES/UFRN
# =============================================================================
# Modelos integrados:
#   [Causal 1]  causal-gpt2-pt  → pierreguillou/gpt2-small-portuguese + LoRA
#   [Causal 2]  causal-bloom    → bigscience/bloom-560m + LoRA
#   [Seq2Seq 1] seq2seq-ptt5    → unicamp-dl/ptt5-base-portuguese-vocab + LoRA
#   [Seq2Seq 2] seq2seq-mt5     → google/mt5-small + LoRA
#
# Diferença fundamental entre os tipos de pipeline:
#   Causal   → pipeline("text-generation")   | prompt = "Instruction: ...\nOutput:"
#   Seq2Seq  → pipeline("text2text-generation") | input = "Instruction: ..."
# =============================================================================

import os
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Importações HuggingFace — ambos os tipos de modelo
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,   # Seq2Seq — encoder-decoder (T5, mT5)
    AutoTokenizer,
    pipeline,
)
from peft import PeftModel   # Carrega adaptadores LoRA sobre o modelo base

import torch

# =============================================================================
# CONFIGURAÇÃO DE LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# INSTÂNCIA FASTAPI
# =============================================================================
app = FastAPI(
    title="RAG + LoRA — API RESTful (4 Modelos)",
    description=(
        "API para comparação de 4 modelos fine-tunados com LoRA sobre o manual "
        "técnico da Grade Hidráulica GH (Marchesan). "
        "Inclui 2 modelos Causais (GPT-2 PT e BLOOM-560M) e "
        "2 modelos Seq2Seq (ptt5-base e mT5-small)."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DICIONÁRIO GLOBAL DE MODELOS
# Cada entrada armazena:
#   - pipeline  : objeto pipeline HuggingFace pronto para inferência
#   - tokenizer : tokenizador do modelo (para contar tokens)
#   - tipo      : "causal" ou "seq2seq" — determina como extrair a resposta
# =============================================================================
MODELS: dict = {}

# =============================================================================
# CAMINHOS DOS MODELOS SALVOS
# Ajuste os caminhos conforme onde você salvou os modelos treinados
# =============================================================================
PATHS = {
    "causal-gpt2-pt": "./lora_causal_model_1",
    "causal-bloom":   "./lora_causal_model_2",
    "seq2seq-ptt5":   "./lora_seq2seq_model_1",
    "seq2seq-mt5":    "./lora_seq2seq_model_2",
}

BASE_MODELS = {
    "causal-gpt2-pt": "pierreguillou/gpt2-small-portuguese",
    "causal-bloom":   "bigscience/bloom-560m",
    "seq2seq-ptt5":   "unicamp-dl/ptt5-base-portuguese-vocab",
    "seq2seq-mt5":    "google/mt5-small",
}

# =============================================================================
# FUNÇÕES DE CARREGAMENTO — MODELOS CAUSAIS
# =============================================================================

def carregar_causal(model_key: str) -> dict:
    """
    Carrega um modelo Causal fine-tunado com LoRA.

    Fluxo:
      1. Carrega o modelo base (AutoModelForCausalLM)
      2. Aplica os adaptadores LoRA (PeftModel.from_pretrained)
      3. Cria o pipeline "text-generation"

    O pipeline causal recebe o prompt completo e gera uma continuação.
    A resposta é extraída removendo o prompt do texto gerado.
    """
    model_path  = PATHS[model_key]
    base_name   = BASE_MODELS[model_key]

    logger.info(f"[{model_key}] Carregando modelo CAUSAL de '{model_path}'...")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modelo '{model_key}' não encontrado em '{model_path}'. "
            f"Treine e salve o modelo antes de iniciar a API."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Carrega base + adaptadores LoRA
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.float32
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    device = 0 if torch.cuda.is_available() else -1

    pipe = pipeline(
        "text-generation",        # Causal: gera continuação do prompt
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    logger.info(f"[{model_key}] ✓ Modelo CAUSAL carregado!")
    return {"pipeline": pipe, "tokenizer": tokenizer, "tipo": "causal"}


def carregar_causal_gpt2_pt() -> dict:
    """Causal 1 — gpt2-small-portuguese + LoRA"""
    return carregar_causal("causal-gpt2-pt")


def carregar_causal_bloom() -> dict:
    """Causal 2 — bloom-560m + LoRA"""
    return carregar_causal("causal-bloom")


# =============================================================================
# FUNÇÕES DE CARREGAMENTO — MODELOS SEQ2SEQ
# =============================================================================

def carregar_seq2seq(model_key: str) -> dict:
    """
    Carrega um modelo Seq2Seq fine-tunado com LoRA.

    Diferenças em relação ao causal:
      - AutoModelForSeq2SeqLM em vez de AutoModelForCausalLM
      - Pipeline "text2text-generation" em vez de "text-generation"
      - A resposta já vem separada do input (encoder → decoder)
        sem necessidade de remover o prompt manualmente

    Fluxo:
      1. Carrega o modelo base Seq2Seq (T5/mT5)
      2. Aplica os adaptadores LoRA
      3. Cria o pipeline "text2text-generation"
    """
    model_path = PATHS[model_key]
    base_name  = BASE_MODELS[model_key]

    logger.info(f"[{model_key}] Carregando modelo SEQ2SEQ de '{model_path}'...")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modelo '{model_key}' não encontrado em '{model_path}'. "
            f"Treine e salve o modelo antes de iniciar a API."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Carrega base Seq2Seq + adaptadores LoRA
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_name,
        torch_dtype=torch.float32
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    device = 0 if torch.cuda.is_available() else -1

    pipe = pipeline(
        "text2text-generation",   # Seq2Seq: gera sequência independente do input
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    logger.info(f"[{model_key}] ✓ Modelo SEQ2SEQ carregado!")
    return {"pipeline": pipe, "tokenizer": tokenizer, "tipo": "seq2seq"}


def carregar_seq2seq_ptt5() -> dict:
    """Seq2Seq 1 — ptt5-base-portuguese-vocab + LoRA"""
    return carregar_seq2seq("seq2seq-ptt5")


def carregar_seq2seq_mt5() -> dict:
    """Seq2Seq 2 — mt5-small + LoRA"""
    return carregar_seq2seq("seq2seq-mt5")


# =============================================================================
# EVENTO DE INICIALIZAÇÃO
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Carrega os 4 modelos na inicialização do servidor.
    Falhas individuais são logadas mas não impedem os outros de carregar.
    """
    global MODELS

    logger.info("=" * 65)
    logger.info("  INICIANDO API — Carregando 4 modelos LoRA...")
    logger.info("=" * 65)

    loaders = {
        "causal-gpt2-pt": carregar_causal_gpt2_pt,
        "causal-bloom":   carregar_causal_bloom,
        "seq2seq-ptt5":   carregar_seq2seq_ptt5,
        "seq2seq-mt5":    carregar_seq2seq_mt5,
    }

    for key, loader in loaders.items():
        try:
            MODELS[key] = loader()
        except Exception as e:
            logger.error(f"[{key}] ✗ Falha ao carregar: {e}")

    logger.info("=" * 65)
    logger.info(f"  ✓ {len(MODELS)}/4 modelo(s) disponível(is): {list(MODELS.keys())}")
    logger.info("=" * 65)


# =============================================================================
# SCHEMAS PYDANTIC
# =============================================================================

class ChatRequest(BaseModel):
    """
    Schema da requisição de chat.

    Campos:
      - modelo     : chave do modelo (ex: "causal-gpt2-pt")
      - mensagem   : instrução/pergunta do usuário
      - max_tokens : máximo de tokens a gerar (padrão: 128)
      - temperatura: criatividade da geração (padrão: 0.7)
                     Ignorado em Seq2Seq com beam search
    """
    modelo: str
    mensagem: str
    max_tokens: Optional[int] = 128
    temperatura: Optional[float] = 0.7


class ChatResponse(BaseModel):
    """Schema da resposta de chat."""
    resposta: str
    modelo: str
    tipo_modelo: str     # "causal" ou "seq2seq"
    tokens_gerados: int


# =============================================================================
# INFORMAÇÕES DOS MODELOS (para /modelos)
# =============================================================================
MODELOS_INFO = {
    "causal-gpt2-pt": {
        "id":        "causal-gpt2-pt",
        "nome":      "GPT-2 Portuguese (Causal)",
        "descricao": (
            "gpt2-small-portuguese fine-tunado com LoRA. "
            "Modelo causal decoder-only pré-treinado em PT-BR. "
            "Vocabulário especializado para português (~124M parâmetros)."
        ),
        "tipo":      "causal",
        "base":      "pierreguillou/gpt2-small-portuguese",
    },
    "causal-bloom": {
        "id":        "causal-bloom",
        "nome":      "BLOOM-560M (Causal)",
        "descricao": (
            "bloom-560m fine-tunado com LoRA. "
            "Modelo causal multilingual treinado em 46 línguas. "
            "Maior capacidade (~560M parâmetros), vocabulário de 250k tokens."
        ),
        "tipo":      "causal",
        "base":      "bigscience/bloom-560m",
    },
    "seq2seq-ptt5": {
        "id":        "seq2seq-ptt5",
        "nome":      "ptt5-base Portuguese (Seq2Seq)",
        "descricao": (
            "ptt5-base-portuguese-vocab fine-tunado com LoRA. "
            "Arquitetura T5 encoder-decoder adaptada ao PT-BR pela Unicamp. "
            "Vocabulário de 32k tokens específicos para português (~248M parâmetros)."
        ),
        "tipo":      "seq2seq",
        "base":      "unicamp-dl/ptt5-base-portuguese-vocab",
    },
    "seq2seq-mt5": {
        "id":        "seq2seq-mt5",
        "nome":      "mT5-small Multilingual (Seq2Seq)",
        "descricao": (
            "mt5-small fine-tunado com LoRA. "
            "Arquitetura mT5 encoder-decoder multilingual (101 línguas). "
            "Vocabulário de 250k tokens sem fine-tuning supervisionado prévio (~300M parâmetros)."
        ),
        "tipo":      "seq2seq",
        "base":      "google/mt5-small",
    },
}

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/modelos", response_class=JSONResponse)
async def listar_modelos():
    """
    GET /modelos

    Retorna todos os modelos disponíveis com id, nome, descrição e tipo.
    O front-end usa este endpoint para popular o dropdown de seleção.
    Apenas modelos carregados com sucesso são retornados.
    """
    disponiveis = [
        info for key, info in MODELOS_INFO.items()
        if key in MODELS
    ]
    return {"modelos": disponiveis, "total": len(disponiveis)}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    POST /chat

    Endpoint principal de inferência. Trata corretamente:
      - Modelos CAUSAIS: formata prompt com "Instruction: ...\\nOutput:"
        e extrai a resposta removendo o prompt do texto gerado.
      - Modelos SEQ2SEQ: passa apenas a instrução ao encoder;
        a resposta do decoder já vem separada.

    Corpo da requisição:
    {
        "modelo": "seq2seq-ptt5",
        "mensagem": "Qual a velocidade de trabalho recomendada?",
        "max_tokens": 128,
        "temperatura": 0.7
    }
    """
    # --- Validações ---
    if request.modelo not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Modelo '{request.modelo}' não encontrado. "
                f"Disponíveis: {list(MODELS.keys())}"
            )
        )

    if not request.mensagem.strip():
        raise HTTPException(
            status_code=400,
            detail="A mensagem não pode ser vazia."
        )

    entrada = request.mensagem.strip()
    modelo_info = MODELS[request.modelo]
    pipe        = modelo_info["pipeline"]
    tokenizer   = modelo_info["tokenizer"]
    tipo        = modelo_info["tipo"]

    logger.info(
        f"[CHAT] modelo='{request.modelo}' tipo='{tipo}' "
        f"mensagem='{entrada[:60]}...'"
    )

    try:
        # ---------------------------------------------------------------
        # GERAÇÃO — MODELO CAUSAL
        # Prompt inclui "Output:" para sinalizar onde começa a resposta.
        # O texto gerado inclui o prompt inteiro; removemos depois.
        # ---------------------------------------------------------------
        if tipo == "causal":
            prompt = f"Instruction: {entrada}\nOutput:"

            resultado = pipe(
                prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperatura,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
            )

            texto_completo = resultado[0]["generated_text"]
            # Remove o prompt — mantém apenas o que veio após "Output:"
            if "Output:" in texto_completo:
                resposta = texto_completo.split("Output:")[-1].strip()
            else:
                resposta = texto_completo[len(prompt):].strip()

        # ---------------------------------------------------------------
        # GERAÇÃO — MODELO SEQ2SEQ
        # Apenas a instrução é passada; o decoder gera a resposta do zero.
        # O resultado já contém APENAS a resposta — sem remover prompt.
        # ---------------------------------------------------------------
        else:  # seq2seq
            input_text = f"Instruction: {entrada}"

            resultado = pipe(
                input_text,
                max_new_tokens=request.max_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

            # Para Seq2Seq, generated_text já é apenas a resposta gerada
            resposta = resultado[0]["generated_text"].strip()

        # --- Resposta vazia ---
        if not resposta:
            resposta = "[O modelo não gerou resposta. Tente reformular a pergunta.]"

        tokens_gerados = len(tokenizer.encode(resposta))
        logger.info(f"  ✓ Resposta gerada: {tokens_gerados} tokens")

        return ChatResponse(
            resposta=resposta,
            modelo=request.modelo,
            tipo_modelo=tipo,
            tokens_gerados=tokens_gerados,
        )

    except Exception as e:
        logger.error(f"[CHAT] Erro na geração com '{request.modelo}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao gerar resposta: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """
    GET /health

    Verifica o status do servidor e quais modelos estão carregados.
    Retorna tipo e modelo base de cada um para debugging.
    """
    modelos_status = {}
    for key in MODELOS_INFO:
        if key in MODELS:
            modelos_status[key] = {
                "status":     "carregado",
                "tipo":       MODELS[key]["tipo"],
                "base":       MODELOS_INFO[key]["base"],
            }
        else:
            modelos_status[key] = {
                "status":     "não carregado",
                "tipo":       MODELOS_INFO[key]["tipo"],
                "base":       MODELOS_INFO[key]["base"],
            }

    return {
        "status":             "ok",
        "modelos_carregados": list(MODELS.keys()),
        "quantidade":         len(MODELS),
        "detalhes":           modelos_status,
        "gpu_disponivel":     torch.cuda.is_available(),
    }


# =============================================================================
# FRONT-END ESTÁTICO
# =============================================================================
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a página principal do chat."""
    html_path = os.path.join("static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# =============================================================================
# PONTO DE ENTRADA
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False   # False em produção — reload=True causa duplo carregamento de modelos
    )
