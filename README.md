<<<<<<< HEAD
# 🧪 Laboratório: Clone do ChatGPT com FastAPI + LoRA

Interface web minimalista para interagir com modelos de linguagem HuggingFace
(modelo base e fine-tunado com LoRA) via API REST construída com FastAPI.

---

## Estrutura do Projeto

```
chatgpt-clone/
│
├── main.py               ← Backend FastAPI (API + servidor de arquivos)
├── requirements.txt      ← Dependências Python
├── README.md             ← Este arquivo
│
├── static/
│   └── index.html        ← Front-end (HTML + CSS + JS puro)
│
└── lora_finetuned_model/ ← (coloque aqui os adaptadores LoRA)
    └── distilgpt2_tokenizer/  ← (coloque aqui o tokenizador salvo)
```

---

## Instalação

```bash
# 1. Clone ou copie o projeto
cd chatgpt-clone

# 2. Crie um ambiente virtual (recomendado)
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# 3. Instale as dependências
pip install -r requirements.txt
```

---

## Executando o Servidor

```bash
# Opção 1: direto pelo Python
python main.py

# Opção 2: via uvicorn (recomendado)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Acesse: **http://localhost:8000**

---

## Usando o Modelo Fine-tunado (LoRA)

Coloque os arquivos do seu modelo fine-tunado no diretório raiz:

```
chatgpt-clone/
├── lora_finetuned_model/    ← adaptadores LoRA + config
│   ├── config.json
│   ├── adapter_model.bin
│   └── ...
└── distilgpt2_tokenizer/    ← tokenizador salvo localmente
    ├── tokenizer.json
    └── ...
```

Se os diretórios não existirem, a aplicação usa o modelo base como **fallback** automático.

---

## Endpoints da API

| Método | Rota       | Descrição                          |
|--------|------------|------------------------------------|
| GET    | `/`        | Interface web do chat              |
| GET    | `/modelos` | Lista modelos disponíveis          |
| POST   | `/chat`    | Envia mensagem, recebe resposta    |
| GET    | `/health`  | Status do servidor e modelos       |
| GET    | `/docs`    | Documentação interativa (Swagger)  |

### Exemplo: POST /chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "modelo": "distilgpt2-base",
    "mensagem": "What is machine learning?",
    "max_tokens": 100,
    "temperatura": 0.7
  }'
```

---

## Parâmetros de Geração

| Parâmetro    | Descrição                                              | Padrão |
|--------------|--------------------------------------------------------|--------|
| `temperatura`| Aleatoriedade da geração (0.1 = preciso, 1.5 = criativo)| 0.7   |
| `max_tokens` | Máximo de tokens novos gerados                         | 150    |

---

## Documentação Interativa

Com o servidor rodando, acesse **http://localhost:8000/docs** para a interface
Swagger gerada automaticamente pelo FastAPI.
=======
# avaliacao-02
>>>>>>> b69f651fa6eb8e629ef9eb711a214acbf036f372
