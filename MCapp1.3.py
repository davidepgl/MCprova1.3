import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Professional Portfolio Stress-Test", layout="wide")

# --- FUNZIONE DOWNLOAD DATI (VERSIONE ROBUSTA) ---
@st.cache_data(ttl=86400)
def get_live_market_data():
    try:
        tickers = ["SPY", "TLT"]
        df = yf.download(tickers, start="2002-01-01", interval="1mo", auto_adjust=True)
        # Gestione Multi-Index di Yahoo Finance
        if isinstance(df.columns, pd.MultiIndex):
            data = df['Close']
        else:
            data = df
        returns = data.ffill().pct_change().dropna()
        # Assicuriamoci che le colonne siano identificate correttamente
        return returns[["SPY", "TLT"]]
    except Exception as e:
        st.error(f"Errore connessione dati reali: {e}")
        return None

# Caricamento iniziale
live_returns = get_live_market_data()

# --- MOTORE MONTE CARLO ---
def run_simulation(capitale, prelievo_pct, equity_pct, anni, ter, n_sim, extra_expenses, mode, params=None):
    mesi = int(anni * 12)
    prelievo_mensile = (capitale * (prelievo_pct / 100)) / 12
    costi_mensili = (ter / 100) / 12
    
    # Spese extra
    spese_pianificate = np.zeros(mesi + 1)
    for _, row in extra_expenses.iterrows():
        m = int(row['Anno'] * 12)
        if 0 <= m <= mesi:
            spese_pianificate[m] += row['Importo (€)']

    # Generazione Rendimenti
    if mode == "Bootstrap (Dati Reali Live)" and live_returns is not None:
        idx = np.random.randint(0, len(live_returns), size=(mesi, n_sim))
        h_spy = live_returns['SPY'].values[idx]
        h_tlt = live_returns['TLT'].values[idx]
    else:
        # Modalità Parametrica (Gaussiana)
        m_spy = params['m_spy'] / 12
        s_spy = params['s_spy'] / np.sqrt(12)
        m_tlt = params['m_tlt'] / 12
        s_tlt = params['s_tlt'] / np.sqrt(12)
        corr = params['corr']
        
        cov_val = corr * s_spy * s_tlt
        cov_matrix = [[s_spy**2, cov_val], [cov_val, s_tlt**2]]
        
        rets = np.random.multivariate_normal([m_spy, m_tlt], cov_matrix, size=(mesi, n_sim))
        h_spy, h_tlt = rets[:,:,0], rets[:,:,1]

    port_returns = (h_spy * equity_pct) + (h_tlt * (1 - equity_pct))
    percorsi = np.zeros((mesi + 1, n_sim))
    percorsi[0] = capitale
    
    for t in range(mesi):
        val = percorsi[t] * (1 + port_returns[t] - costi_mensili)
        val = val - prelievo_mensile - spese_pianificate[t+1]
        val[val < 0] = 0
        percorsi[t+1] = val
    return percorsi

# --- INTERFACCIA UTENTE ---
st.title("🏛️ Simulatore Monte Carlo High-Performance")

with st.sidebar:
    st.header("1. Motore di Simulazione")
    sim_mode = st.radio("Sorgente Rendimenti:", ["Bootstrap (Dati Reali Live)", "Parametrica (Gaussiana Custom)"])
    
    param_dict = {}
    if sim_mode == "Parametrica (Gaussiana Custom)":
        st.subheader("⚙️ Ipotesi Asset Class")
        st.markdown("**Azionario (Equity)**")
        m_spy = st.number_input("$\mu$ reale annuo %", value=7.0) / 100
        s_spy = st.number_input("$\sigma$ annuo %", value=18.0) / 100
        st.markdown("**Obbligazionario (Bond)**")
        m_tlt = st.number_input("$\mu$ reale annuo % ", value=2.0) / 100
        s_tlt = st.number_input("$\sigma$ annuo % ", value=7.0) / 100
        corr = st.slider("Correlazione (Azioni/Bond)", -1.0, 1.0, 0.0)
        param_dict = {'m_spy': m_spy, 's_spy': s_spy, 'm_tlt': m_tlt, 's_tlt': s_tlt, 'corr': corr}
    
    st.divider()
    st.header("2. Input Portafoglio")
    cap = st.number_input("Capitale Iniziale (€)", value=1000000, step=50000)
    prel = st.slider("Prelievo Annuo Lordo (%)", 0.0, 15.0, 4.0)
    eq = st.slider("Esposizione Azionaria", 0.0, 1.0, 0.6)
    yrs = st.slider("Anni Proiezione", 1, 50, 30)
    ter = st.slider("Costi (TER) %", 0.0, 5.0, 1.5)
    sim = st.selectbox("Precisione (N. Scenari)", [10000, 50000, 100000], index=1)
    
    st.header("3. Uscite Straordinarie")
    df_extra = pd.DataFrame([{"Anno": 10, "Importo (€)": 0.0}])
    edited_df = st.data_editor(df_extra, num_rows="dynamic", use_container_width=True)
    
    btn = st.button("AVVIA SIMULAZIONE", type="primary", use_container_width=True)

if btn:
    with st.spinner('Calcolo in corso...'):
        dati = run_simulation(cap, prel, eq, yrs, ter, sim, edited_df, sim_mode, param_dict)
    
    p_levels = [5, 10, 25, 50, 75, 90, 95]
    pct = {p: np.percentile(dati, p, axis=1) for p in p_levels}
    successo = np.mean(dati[-1, :] > 0) * 100

    # --- RIQUADRI KPI ---
    prel_ann_N = (cap * (prel / 100)) * 0.74
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probabilità Successo", f"{successo:.1f}%")
    c2.metric("Annuo Netto (26%)", f"€ {prel_ann_N:,.0f}")
    c3.metric("Mensile Netto", f"€ {prel_ann_N/12:,.0f}")
    c4.metric("Scenari Elaborati", f"{sim:,}")

    # --- FEEDBACK ---
    if successo > 85:
        st.success("✅ Il piano finanziario è solido per l'orizzonte scelto.")
    elif successo > 70:
        st.warning("⚠️ Attenzione: Il prelievo è aggressivo. Margini di sicurezza ridotti.")
    else:
        st.error("🚨 Rischio esaurimento elevato. Necessario ridurre prelievi o costi.")

    # --- SINTESI 3 SCENARI ---
    st.subheader("📊 Analisi Scenari Chiave")
    s1, s2, s3 = st.columns(3)
    scenarios = {10: "Stress Test (P10)", 50: "Scenario Base (P50)", 90: "Scenario Ottimista (P90)"}
    
    for col, p in zip([s1, s2, s3], [10, 50, 90]):
        with col:
            st.markdown(f"### {scenarios[p]}")
            cap_fin = pct[p][-1]
            st.write(f"Capitale Finale: **€ {cap_fin:,.0f}**")
            if cap_fin == 0:
                anno_es = np.where(pct[p] == 0)[0][0] // 12
                st.write(f"Esaurimento: 🔴 **Anno {anno_es}**")
            else: st.write("Esaurimento: ✅ Mai")
            st.write(f"Minimo toccato: € {np.min(pct[p]):,.0f}")
            st.write(f"Tot. Prelevato: € {(cap*(prel/100)*yrs) + edited_df['Importo (€)'].sum():,.0f}")

    # --- FAN CHART ---
    st.subheader("📈 Evoluzione Probabilistica (Fan Chart)")
    fig, ax = plt.subplots(figsize=(12, 5))
    t_range = np.arange(yrs * 12 + 1)
    ax.fill_between(t_range, pct[5], pct[95], color='royalblue', alpha=0.1, label='Intervallo 90% (P5-P95)')
    ax.fill_between(t_range, pct[25], pct[75], color='royalblue', alpha=0.3, label='Intervallo 50% (P25-P75)')
    ax.plot(t_range, pct[50], color='navy', linewidth=2, label='Mediana (P50)')
    ax.plot(t_range, pct[10], color='red', linestyle='--', label='Stress Test (P10)')
    
    # Annotazioni Spese
    for _, row in edited_df.iterrows():
        mese = int(row['Anno'] * 12)
        if row['Importo (€)'] > 0 and mese < len(pct[50]):
            ax.annotate('OUT', xy=(mese, pct[50][mese]), xytext=(0, 10), textcoords='offset points', color='red', fontsize=8, arrowprops=dict(arrowstyle='->', color='red'))

    ax.set_ylabel("Capitale (€)")
    ax.legend(loc='upper left')
    st.pyplot(fig)

    # --- TABELLA TEMPORALE ---
    st.subheader("📅 Proiezione Temporale Percentili")
    step = 5 if yrs > 15 else 2
    idx_annuali = np.arange(0, (yrs * 12) + 1, step * 12)
    df_tab = pd.DataFrame({f"P{p}": pct[p][idx_annuali] for p in p_levels}, index=[f"Anno {i//12}" for i in idx_annuali])
    st.dataframe(df_tab.style.format("{:,.0f}"), use_container_width=True)