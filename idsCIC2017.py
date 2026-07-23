import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

# ----- CONFIGURAÇÕES -----
caminho = r'C:\IC\experimentoIDS\mods_rotulados\\'
colunas_para_remover = ['id', 'expiration_id', 'src_ip', 'src_mac','src_oui', 'src_port', 'dst_ip', 'dst_mac', 'dst_oui', 'dst_port','protocol', 'ip_version', 'vlan_id', 'tunnel_id','bidirectional_first_seen_ms','bidirectional_last_seen_ms','bidirectional_duration_ms', 'src2dst_first_seen_ms','src2dst_last_seen_ms', 'src2dst_duration_ms','dst2src_first_seen_ms', 'dst2src_last_seen_ms','dst2src_duration_ms', 'bidirectional_min_piat_ms','bidirectional_mean_piat_ms', 'bidirectional_stddev_piat_ms','bidirectional_max_piat_ms', 'src2dst_min_piat_ms','src2dst_mean_piat_ms','src2dst_stddev_piat_ms', 'src2dst_max_piat_ms','dst2src_min_piat_ms', 'dst2src_mean_piat_ms', 'dst2src_stddev_piat_ms','dst2src_max_piat_ms','datetime']

df1_raw = pd.read_csv(caminho + 'base_Mod1_rotulada.csv')
df2_raw = pd.read_csv(caminho + 'base_Mod2_rotulada.csv')
df3_raw = pd.read_csv(caminho + 'base_Mod3_rotulada.csv')
C2017 = pd.read_csv(r'C:\IC\experimentoIDS\CIC2017-nfstream-target.csv', sep=None, engine='python')

print("\n=== VERIFICAÇÃO DAS BASES ===")
print("Base 1:"); print(df1_raw['label'].value_counts())
print("Base 2:"); print(df2_raw['label'].value_counts())
print("Base 3:"); print(df3_raw['label'].value_counts())

lista_ataques = ["DoS-Slowhttptest", "DoS-GoldenEye", "DoS-Hulk", "DoS-slowloris", "FTP-Patator", "SSH-Patator", "NMAP-Completo"]

def preparar_ataque(df_raw, nome_atq):
    # Filtra benigno + ataque, cria target, limpa e retorna as colunas numéricas
    temp = df_raw[(df_raw['label'] == 'benign') | (df_raw['label'] == nome_atq)].copy()
    if temp.empty or len(temp['label'].unique()) < 2:
        return pd.DataFrame()
    temp['target'] = (temp['label'] == nome_atq).astype(int)
    # Seleciona numéricas
    temp = temp.select_dtypes(include=['number'])
    temp = temp.drop(columns=colunas_para_remover, errors='ignore')
    temp = temp.replace([np.inf, -np.inf], np.nan).dropna()
    return temp

def alinhar_colunas(df_tr, df_ts):
    # Garante que treino e teste tenham as mesmas colunas
    cols_tr = set(df_tr.columns) - {'target'}
    cols_ts = set(df_ts.columns) - {'target'}
    cols_comuns = sorted(cols_tr & cols_ts)
    return df_tr[cols_comuns + ['target']], df_ts[cols_comuns + ['target']]

def treinar_e_avaliar(df_tr, df_ts, nome_cenario):
    """Treina RF e retorna dict com métricas."""
    if df_tr.empty or df_ts.empty:
        return None
    if len(df_tr['target'].unique()) < 2:
        print(f"   [AVISO] Cenário {nome_cenario}: treino com apenas 1 classe. Pulando.")
        return None

    df_tr, df_ts = alinhar_colunas(df_tr, df_ts)

    X_tr, y_tr = df_tr.drop(columns=['target']), df_tr['target']
    X_ts, y_ts = df_ts.drop(columns=['target']), df_ts['target']

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    y_pred = rf.predict(X_ts)
    y_prob = rf.predict_proba(X_ts)
    prob = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob[:, 0]

    return {
        'cenarios': nome_cenario,
        'AUC': round(roc_auc_score(y_ts, prob), 4),
        'F1-score': round(f1_score(y_ts, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_ts, y_pred, zero_division=0), 4),
        'precisao': round(precision_score(y_ts, y_pred, zero_division=0), 4)
    }

# ----- EXECUÇÃO -----
with pd.ExcelWriter('Experimento3_CIC2017.xlsx', engine='xlsxwriter') as writer:
    for ataque in lista_ataques:
        print(f"\nProcessando: {ataque}")

        f1 = preparar_ataque(df1_raw, ataque)
        f2 = preparar_ataque(df2_raw, ataque)
        f3 = preparar_ataque(df3_raw, ataque)
        fcic = preparar_ataque(C2017, ataque)

        if f1.empty and f2.empty and f3.empty:
            print(f"   [AVISO] Ataque '{ataque}' não encontrado em nenhuma base local. Pulando.")
            continue

        resultados = []

        # ── Cenário BASE: 70% treino / 30% teste (todas as bases locais juntas, aleatório) ──
        df_all = pd.concat([d for d in [f1, f2, f3] if not d.empty])
        if len(df_all['target'].unique()) > 1:
            df_tr_base, df_ts_base = train_test_split(
                df_all, test_size=0.3, random_state=42, stratify=df_all['target']
            )
            res = treinar_e_avaliar(df_tr_base, df_ts_base, 'base')
            if res:
                resultados.append(res)

        cenarios = [
            ([f1, f2], [f3],      "c1"),
            ([f1, f3], [f2],      "c2"),
            ([f2, f3], [f1],      "c3"),
            ([f1],     [f2, f3],  "c4"),
            ([f2],     [f1, f3],  "c5"),
            ([f3],     [f1, f2],  "c6"),
        ]

        # Executa c1 até c6
        for tr_l, ts_l, nome_c in cenarios:
            df_tr = pd.concat([d for d in tr_l if not d.empty])
            df_ts = pd.concat([d for d in ts_l if not d.empty])
            
            res = treinar_e_avaliar(df_tr, df_ts, nome_c)
            if res:
                resultados.append(res)

        # ── Cenários c7 e c8 Originais (Base CIC2017 Padrão) ──
        if not fcic.empty and len(fcic['target'].unique()) > 1:
            # c7 original
            res_c7 = treinar_e_avaliar(fcic, df_all, "c7")
            if res_c7: resultados.append(res_c7)
            
            # c8 original
            res_c8 = treinar_e_avaliar(df_all, fcic, "c8")
            if res_c8: resultados.append(res_c8)
            
            # ──  c9 e c10 (Com Fluxo Normal Isolado do CIC) ──
            cic_normal = fcic[fcic['target'] == 0].copy()
            cic_ataque = fcic[fcic['target'] == 1].copy()

            #Separa apenas a parte de ATAQUES das bases locais juntas (remover o normal local)
            local_ataques = df_all[df_all['target'] == 1].copy()

            base1 = pd.concat([cic_normal, cic_ataque], axis=0)  # CIC-normal + CIC-ataque
            base2 = pd.concat([cic_normal, local_ataques], axis=0) # CIC-normal + local-ataques

            # c9: Treina na base1 (CIC puro) -> Testa na base2 (Normal CIC + Ataque Local)
            res_c9 = treinar_e_avaliar(base1, base2, "c9")
            if res_c9: resultados.append(res_c9)

            # c10: Treina na base2 (Normal CIC + Ataque Local) -> Testa na base1 (CIC puro)
            res_c10 = treinar_e_avaliar(base2, base1, "c10")
            if res_c10: resultados.append(res_c10)

        else:
            print(f"   [AVISO] Ataque '{ataque}' ausente ou sem classes no CIC2017. Pulando c7, c8, c9 e c10.")

        # Gravando todos os resultados na planilha
        if resultados:
            pd.DataFrame(resultados).to_excel(writer, sheet_name=ataque[:31], index=False)
            print(f"   [SUCESSO] Aba '{ataque}' gravada com todos os cenários (base até c10).")

print("\nFim do experimento!")