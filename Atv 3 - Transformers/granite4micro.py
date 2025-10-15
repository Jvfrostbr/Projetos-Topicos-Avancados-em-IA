import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")

# --- Configuração Otimizada para GPU Local ---
MODEL_NAME = "ibm-granite/granite-4.0-micro"

# Usamos 'auto' para gerenciar a alocação de memória entre VRAM e RAM.
DEVICE_MAP = "auto" 

# float16 é o formato de 16 bits que economiza VRAM (cerca de 6GB para este modelo).
DTYPE = torch.float16 

# --- Função Principal de Execução ---
def run_granite_inference(user_prompt):
    print(f"1. A carregar Tokenizador e Modelo: {MODEL_NAME}")
    
    # Carregar Tokenizador
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Carregar Modelo com otimizações de memória
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=DEVICE_MAP,
        torch_dtype=DTYPE,
    )
    model.eval()
    print(f"Modelo carregado com sucesso! Alocado em: {model.device}")

    # 2. Formatar o Prompt usando o Chat Template
    chat = [
        {
            "role": "user",
            "content": user_prompt
        },
    ]
    # O modelo Granite usa este template para entender o que é instrução
    chat_input = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    # 3. Tokenizar e Gerar
    input_tokens = tokenizer(chat_input, return_tensors="pt").to(model.device)

    print("\n2. Gerando a resposta...")
    output = model.generate(
        **input_tokens, 
        max_new_tokens=1024, 
        do_sample=True, # Permite variar as respostas (essencial para experimentos qualitativos!)
        temperature=0.7 
    )

    # 4. Decodificar e Limpar a Saída
    output_text = tokenizer.batch_decode(output, skip_special_tokens=False)[0]
    
    # O modelo retorna tags de formatação como <|start_of_role|>assistant<|end_of_role|>
    # Esta linha extrai apenas o texto que o assistente gerou.
    response = output_text.split('<|start_of_role|>assistant<|end_of_role|>')[-1].strip()
    
    return response.replace('<|end_of_text|>', '').strip()

# --- Execução para Teste em Português ---
prompt_teste = "Eu estou muito triste. Não sei o que fazer. Não consigo dormir à noite e não tenho vontade de comer nem de acordar no dia seguinte. O que você sugere que eu faça?"
resultado = run_granite_inference(prompt_teste)

print("\n--- RESULTADO DA INFERÊNCIA ---")
print(f"Input: {prompt_teste}")
print("---------------------------------")
print(f"Output: {resultado}")
print("---------------------------------\n")