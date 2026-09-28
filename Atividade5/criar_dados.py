import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Cria o grafico

np.random.seed(42)

datas_transacoes = pd.date_range(start='2026-09-01', periods=1000, freq='h')

df_transacoes = pd.DataFrame({
    'id_cliente': np.random.choice(['C100', 'C101', 'C102', 'C103', 'C104'], size=1000),
    'data_transacao': datas_transacoes,
    'valor': np.random.choice(
        [np.nan, 150.0, 3000.0, 7500.0, 12000.0, 50.0], 
        size=1000, 
        p=[0.05, 0.40, 0.30, 0.15, 0.05, 0.05]
    ),
    'estado_cliente': np.random.choice(['SP', 'RJ', 'MG', 'RS'], size=1000),
    'origem': np.random.choice(['web', 'mobile_app', 'atm'], size=1000)
})

df_transacoes = pd.concat([df_transacoes, df_transacoes.iloc[:15]], ignore_index=True)
df_transacoes.to_csv('transacoes.csv', index=False, encoding='latin1')

datas_cotacoes = pd.date_range(start='2026-09-01', end='2026-10-01', freq='D')
variacoes = np.random.normal(loc=0.001, scale=0.015, size=len(datas_cotacoes))
preco_inicial = 5.20
precos = preco_inicial * np.exp(np.cumsum(variacoes))

df_cotacoes = pd.DataFrame({
    'data': datas_cotacoes,
    'cotacao_usd': np.round(precos, 4),
    'volume_negociado': np.random.randint(10000, 500000, size=len(datas_cotacoes))
})
df_cotacoes.loc[5, 'cotacao_usd'] = np.nan
df_cotacoes.loc[18, 'cotacao_usd'] = np.nan
df_cotacoes.to_csv('cotacoes.csv', index=False, encoding='utf-8')

diretorio_atual = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
caminho_transacoes = os.path.join(diretorio_atual, 'transacoes.csv')

df = pd.read_csv(caminho_transacoes, encoding='latin1')

df['valor'] = df.groupby('estado_cliente')['valor'].transform(lambda x: x.fillna(x.median()))

df['plataforma'] = 'Mobile'

df['data_transacao'] = pd.to_datetime(df['data_transacao']).dt.tz_localize('America/Sao_Paulo')

df['dia_semana'] = df['data_transacao'].dt.day_name()
df['mes'] = df['data_transacao'].dt.month_name()

df = df.drop_duplicates(keep='first')

filtro_setembro_premium = (
    (df['data_transacao'].dt.month == 9) & 
    ((df['estado_cliente'] == 'SP') | (df['estado_cliente'] == 'RJ')) & 
    (df['valor'] > 5000.0)
)
df_filtrado = df[filtro_setembro_premium]

risco_dict = {
    'C100': 'Baixo', 
    'C101': 'Alto', 
    'C102': 'Médio', 
    'C103': 'Alto', 
    'C104': 'Baixo'
}

df['nivel_risco'] = df['id_cliente'].map(risco_dict)

pivot_risco = df.pivot_table(
    index='mes', 
    columns='nivel_risco', 
    values='valor', 
    aggfunc='sum', 
    margins=True
)

df['z_score'] = df.groupby('estado_cliente')['valor'].transform(
    lambda x: (x - x.mean()) / x.std()
)
df_anomalias = df[df['z_score'] > 2.5]

df_diario = df.groupby(df['data_transacao'].dt.date)['valor'].sum().reset_index()

df_diario['media_movel_7d'] = df_diario['valor'].rolling(window=7).mean()

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df_diario['data_transacao'], df_diario['valor'], label='Valor Total Diário', color='#1f77b4', alpha=0.6, marker='o', markersize=4)
ax.plot(df_diario['data_transacao'], df_diario['media_movel_7d'], label='Média Móvel (7 dias)', color='#d62728', linewidth=2.5, linestyle='--')

ax.set_ylim(bottom=0)
ax.set_title('Métricas de Desempenho: Transações Diárias vs Tendência Móvel', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Período Temporal (Data)', fontsize=11, labelpad=10)
ax.set_ylabel('Volume Financeiro Total (R$)', fontsize=11, labelpad=10)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
print("Processamento concluído! Exibindo o gráfico...")
plt.show()
