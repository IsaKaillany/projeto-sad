# SIS-EVA: Streamlit app para Análise de Evasão e Previsão de Ingressantes
# Leitura automática de arquivos no diretório: discentes-2023.csv, discentes-2024.csv, discentes-2025.csv

import streamlit as st
import pandas as pd
import numpy as np
import glob
import re
from sklearn.linear_model import LinearRegression
import plotly.express as px
import io

st.set_page_config(layout="wide", page_title="SIS-EVA | Análise de Evasão e Previsão")

st.title("SIS-EVA — Análise de Evasão e Previsão de Ingressantes")
st.markdown(
    "Aplicação automática que lê todos os arquivos `discentes-YYYY.csv` no diretório do app, consolida os dados e gera indicadores de evasão e previsões de ingressantes por curso."
)

# ----------------- Helpers -----------------

def find_csv_files(pattern="discentes-*.csv"):
    files = sorted(glob.glob(pattern))
    return files


def extract_year_from_filename(fname):
    # procura um grupo de 4 dígitos (ano) no nome do arquivo
    m = re.search(r"(19|20)\d{2}", fname)
    return int(m.group(0)) if m else None


@st.cache_data
def load_and_concatenate(files):
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            # tenta separador ;
            df = pd.read_csv(f, sep=';')
        year = extract_year_from_filename(f)
        df['_source_file'] = f
        df['_ingresso_ano'] = year
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)
    return full


def normalize_columns(df):
    # Retorna mapa de colunas principais encontrados
    cols = df.columns.tolist()
    lower_map = {c.lower(): c for c in cols}

    def find(names):
        for n in names:
            if n in lower_map:
                return lower_map[n]
        return None

    col_map = {
        'matricula': find(['matricula','id','registro']),
        'curso': find(['nome_curso','curso','nome do curso','nome_do_curso','curso_nome']),
        'status': find(['status','status_discente','situacao','situacao_discente']),
        'ano_ingresso': find(['ano_ingresso','_ingresso_ano','ano','ano_de_ingresso'])
    }
    return col_map


def classify_status(s):
    s = str(s).upper()
    if any(k in s for k in ['DESIST', 'DESLIG', 'CANCEL', 'TRANC']):
        return 'EVASAO'
    if any(k in s for k in ['CONCLU', 'GRADU']):
        return 'CONCLUIDO'
    if any(k in s for k in ['ATIV', 'MATRIC']):
        return 'ATIVO'
    return 'OUTRO'


# ----------------- Load files -----------------
files = find_csv_files('discentes-*.csv')
if not files:
    st.error("Nenhum arquivo 'discentes-YYYY.csv' encontrado no diretório. Coloque os CSVs (ex.: discentes-2023.csv) na mesma pasta do app.")
    st.stop()

df = load_and_concatenate(files)

st.sidebar.header('Resumo dos dados')
st.sidebar.write(f"Registros carregados: {len(df)}")

# Normalize colunas
col_map = normalize_columns(df)
COL_MAT = col_map['matricula']
COL_CURSO = col_map['curso']
COL_STATUS = col_map['status']
COL_ANO = col_map['ano_ingresso'] if col_map['ano_ingresso'] else '_ingresso_ano'

if COL_CURSO is None:
    st.error('Não foi possível localizar a coluna do nome do curso. Renomeie a coluna para `nome_curso` ou `curso` e tente novamente.')
    st.stop()

# preparar coluna de ano se faltar
if COL_ANO not in df.columns:
    df[COL_ANO] = df['_ingresso_ano']

# normalizar texto
df[COL_CURSO] = df[COL_CURSO].astype(str).str.strip()
if COL_STATUS in df.columns:
    df[COL_STATUS] = df[COL_STATUS].astype(str).str.upper().str.strip()
    df['_status_class'] = df[COL_STATUS].apply(classify_status)
else:
    df['_status_class'] = 'UNKNOWN'

# garantir coluna de matricula para contagens
if COL_MAT not in df.columns:
    df['_mat_temp'] = df.index.astype(str)
    COL_MAT = '_mat_temp'

# ----------------- Métricas por curso -----------------
course_stats = df.groupby(COL_CURSO).apply(lambda g: pd.Series({
    'total_alunos': g[COL_MAT].nunique() if COL_MAT in g.columns else g.shape[0],
    'concluintes': g[g['_status_class']=='CONCLUIDO'][COL_MAT].nunique() if COL_STATUS in df.columns else 0,
    'evasao': g[g['_status_class']=='EVASAO'][COL_MAT].nunique() if COL_STATUS in df.columns else 0,
    'ativos': g[g['_status_class']=='ATIVO'][COL_MAT].nunique() if COL_STATUS in df.columns else 0
}))

course_stats['taxa_evasao'] = (course_stats['evasao'] / course_stats['total_alunos']).fillna(0)
course_stats['taxa_conclusao'] = (course_stats['concluintes'] / course_stats['total_alunos']).fillna(0)
course_stats = course_stats.sort_values('taxa_evasao', ascending=False)

# Classificação simples de risco
def risco_from_taxa(t):
    if t > 0.4:
        return 'Alto'
    if t > 0.2:
        return 'Médio'
    return 'Baixo'

course_stats['risco'] = course_stats['taxa_evasao'].apply(risco_from_taxa)

# ----------------- Previsão de ingressantes por curso (regressão) -----------------
st.header('Resultados: Evasão & Previsão')

st.subheader('Ranking de Cursos por Taxa de Evasão')
st.write('Cursos ordenados pela taxa de evasão (maior primeiro).')
st.dataframe(course_stats.reset_index().rename(columns={COL_CURSO:'curso'}).head(100))

# preparar série histórica de ingressantes por curso e ano
try:
    df[COL_ANO] = pd.to_numeric(df[COL_ANO], errors='coerce')
except Exception:
    pass

entrants = df.groupby([COL_CURSO, COL_ANO]).agg(ingressantes=(COL_MAT, 'nunique')).reset_index()
# remover registros sem ano
entrants = entrants[entrants[COL_ANO].notna()]

# construir previsões
preds = []
for curso, group in entrants.groupby(COL_CURSO):
    years = group[COL_ANO].values.reshape(-1,1)
    values = group['ingressantes'].values
    if len(years) >= 2:
        model = LinearRegression()
        model.fit(years, values)
        next_year = int(group[COL_ANO].max() + 1)
        pred = model.predict(np.array([[next_year]]))[0]
        slope = float(model.coef_[0])
        last_year = int(group[COL_ANO].max())
        last_count = int(group[group[COL_ANO]==group[COL_ANO].max()]['ingressantes'].values[0])
        preds.append({'curso': curso, 'last_year': last_year, 'last_year_count': last_count, 'pred_next_year': max(0, float(pred)), 'slope': slope, 'years_used': len(years)})
    else:
        # insuficiente
        last_year = int(group[COL_ANO].max())
        last_count = int(group[group[COL_ANO]==group[COL_ANO].max()]['ingressantes'].values[0])
        preds.append({'curso': curso, 'last_year': last_year, 'last_year_count': last_count, 'pred_next_year': np.nan, 'slope': np.nan, 'years_used': len(years)})

preds_df = pd.DataFrame(preds)
merged = preds_df.merge(course_stats.reset_index().rename(columns={COL_CURSO:'curso'}), on='curso', how='left')
merged['pred_diff'] = merged['pred_next_year'] - merged['last_year_count']

st.subheader('Previsões de Ingressantes (regressão linear simples)')
st.write('Somente cursos com ao menos 2 anos de histórico têm previsão. A previsão é linear e serve como sinal, não como garantia.')
st.dataframe(merged.sort_values('slope', ascending=False).head(50))

st.write('Top 10 cursos com maior tendência (slope positivo):')
st.table(merged.dropna(subset=['slope']).sort_values('slope', ascending=False).head(10)[['curso','years_used','last_year','last_year_count','pred_next_year','slope']])

# gráfico interativo por curso
st.sidebar.header('Explorar curso')
curso_sel = st.sidebar.selectbox('Selecione um curso para ver série histórica', options=sorted(entrants[COL_CURSO].unique()))
if curso_sel:
    g = entrants[entrants[COL_CURSO]==curso_sel].sort_values(COL_ANO)
    fig = px.line(g, x=COL_ANO, y='ingressantes', title=f'Ingressantes históricos — {curso_sel}', markers=True)
    row = merged[merged['curso']==curso_sel]
    if not row.empty and not np.isnan(row['pred_next_year'].values[0]):
        next_y = int(row['last_year'].values[0]) + 1
        pred_y = row['pred_next_year'].values[0]
        fig.add_scatter(x=[next_y], y=[pred_y], mode='markers+text', name='Predição', text=[f'{pred_y:.0f}'])
    st.plotly_chart(fig, use_container_width=True)

# Painel de detalhes por curso
st.subheader('Painel de Detalhes por Curso')
curso_options = course_stats.reset_index()[COL_CURSO].tolist()
curso_select = st.selectbox('Selecione um curso para detalhes:', options=curso_options)
if curso_select:
    row = course_stats.loc[curso_select]
    st.metric('Total alunos', int(row['total_alunos']))
    st.metric('Taxa de evasão', f"{row['taxa_evasao']*100:.2f}%")
    st.metric('Taxa de conclusão', f"{row['taxa_conclusao']*100:.2f}%")
    st.write('Risco:', row['risco'])
    status_counts = df[df[COL_CURSO]==curso_select]['_status_class'].value_counts().reset_index()
    status_counts.columns = ['status','count']
    fig_pie = px.pie(status_counts, names='status', values='count', title=f'Status dos discentes — {curso_select}')
    st.plotly_chart(fig_pie, use_container_width=True)

# Exportar resultados
st.subheader('Exportar resultados')
export_df = course_stats.reset_index().rename(columns={COL_CURSO:'curso'})
if not merged.empty:
    export_df = export_df.merge(merged[['curso','pred_next_year','slope','years_used','last_year_count','pred_diff']], on='curso', how='left')

buf = io.BytesIO()
export_df.to_excel(buf, index=False)
buf.seek(0)
st.download_button('Exportar relatório (Excel)', data=buf, file_name='sis_eva_relatorio.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

st.markdown('---')
st.caption('Observações: Previsão = regressão linear simples por curso; ajuste thresholds conforme política institucional.')
