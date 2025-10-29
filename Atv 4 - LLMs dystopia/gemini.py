import os
from dotenv import load_dotenv
from crewai import LLM, Agent, Task, Crew, Process

load_dotenv()

# --- Configuração da LLM ---
gemini_llm = LLM(
    model='gemini/gemini-2.5-flash',
    temperature=0.7
)

# --- Agentes ---
teorico = Agent(
    role='Teórico da Conspiração',
    goal='Defender teorias da conspiração de forma convincente e criativa.',
    backstory='Você é um pensador que conecta fatos obscuros e cria narrativas convincentes. Não usa a internet.',
    llm=gemini_llm,
    allow_delegation=False,
    verbose=False,
    memory =True
)

critico = Agent(
    role='Crítico Cético',
    goal='Analisar criticamente as falas do Teórico, apontando falhas e incoerências.',
    backstory='Você é cético e usa raciocínio lógico para questionar teorias improváveis. Não usa a internet.',
    llm=gemini_llm,
    allow_delegation=False,
    verbose=False,
    memory=True
)

supervisor = Agent(
    role='Supervisor Observador',
    goal='Gerar um resumo objetivo do que foi discutido entre o Teórico e o Cético em cada rodada, sem interagir com eles.',
    backstory='Você é um analista imparcial que observa o diálogo e produz um resumo neutro e conciso a cada rodada.',
    llm=gemini_llm,
    allow_delegation=False,
    verbose=True,
    memory = True
)

# --- Função para criar tarefas de conversa ---
def criar_tarefas(num_rodadas):
    tarefas = []
    for i in range(num_rodadas):
        tarefas.append(Task(
            description=f"Rodada {i+1}: O Teórico apresenta uma nova teoria conspiratória sobre o tema em discussão.",
            expected_output="Uma explicação detalhada e criativa da teoria conspiratória.",
            agent=teorico
        ))
        tarefas.append(Task(
            description=f"Rodada {i+1}: O Crítico responde à fala do Teórico, questionando e desafiando suas ideias.",
            expected_output="Uma análise crítica apontando inconsistências e falhas na teoria apresentada.",
            agent=critico
        ))
        tarefas.append(Task(
            description=f"Rodada {i+1}: O Supervisor resume o conteúdo da rodada entre o Teórico e o Cético, destacando os principais pontos e divergências.",
            expected_output="Um resumo neutro e sintético da interação entre os dois agentes nessa rodada.",
            agent=supervisor
        ))
    return tarefas


# --- Configura a simulação ---
NUM_RODADAS = 5
tema = "Controle mental por governos"

tarefas = criar_tarefas(NUM_RODADAS)

conversa = Crew(
    agents=[teorico, critico, supervisor],
    tasks=tarefas,
    process=Process.sequential,
    verbose=True
)

# --- Executa ---
print("### Iniciando conversa simulada ###")
resultado_final = conversa.kickoff(inputs={"tema": tema})

print("\n\n####################################")
print(f"### Resumo Final da Conversa sobre '{tema}' ###")
print("####################################")
print(resultado_final)