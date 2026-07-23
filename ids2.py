import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


# ----- CONFIGURAÇÕES -----

caminho_df_gustavo = r'C:\IC\basesGustavo\\'
df1_hulk=pd.read_csv(caminho_df_gustavo+"df1_DoS_Hulk.csv", sep=';')
df1_slowhttptest=pd.read_csv(caminho_df_gustavo+"df1_DoS_Slowhttptest.csv", sep=';')
df1_slowloris=pd.read_csv(caminho_df_gustavo+"df1_DoS_Slowloris.csv", sep=';')
df1_scan=pd.read_csv(caminho_df_gustavo+"df1_Port_Scan.csv", sep=';')
df1_ftp=pd.read_csv(caminho_df_gustavo+"df1_FTP_Patator.csv", sep=';')
df1_ssh=pd.read_csv(caminho_df_gustavo+"df1_SSH_Patator.csv", sep=';')
df2_hulk=pd.read_csv(caminho_df_gustavo+"df2_DoS_Hulk.csv", sep=';')
df2_slowhttptest=pd.read_csv(caminho_df_gustavo+"df2_DoS_Slowhttptest.csv", sep=';')
df2_slowloris=pd.read_csv(caminho_df_gustavo+"df2_DoS_Slowloris.csv", sep=';')
df2_scan=pd.read_csv(caminho_df_gustavo+"df2_Port_Scan.csv", sep=';')
df2_ftp=pd.read_csv(caminho_df_gustavo+"df2_FTP_Patator.csv", sep=';')
df2_ssh=pd.read_csv(caminho_df_gustavo+"df2_SSH_Patator.csv", sep=';')
dfGustavo_all = pd.concat([d for d in [df1_hulk, df1_slowhttptest, df1_slowloris,df1_scan,df1_ftp,df1_ssh,df2_hulk,df2_slowhttptest,df2_slowloris,df2_scan,df2_ftp,df2_ssh] if not d.empty])

gustavo_col_map = {
    "DoS-Hulk":          ("attack-DoS-Hulk",         [df1_hulk, df2_hulk]),
    "DoS-Slowhttptest":  ("attack-DoS-Slowhttptest", [df1_slowhttptest, df2_slowhttptest]),
    "DoS-slowloris":     ("attack-DoS-slowloris",    [df1_slowloris, df2_slowloris]),
    "FTP-Patator":       ("attack-FTP-Patator",      [df1_ftp, df2_ftp]),
    "SSH-Patator":       ("attack-SSH-Patator",      [df1_ssh, df2_ssh]),
    "NMAP-Completo":     ("attack-Port-Scan",        [df1_scan, df2_scan]),
}


#print(df1_hulk.columns.tolist())



caminho = r'C:\IC\experimentoIDS\mods_rotulados\\'
colunas_para_remover = ['id', 'expiration_id', 'src_ip', 'src_mac','src_oui', 'src_port', 'dst_ip', 'dst_mac', 'dst_oui', 'dst_port','protocol', 'ip_version', 'vlan_id', 'tunnel_id','bidirectional_first_seen_ms','bidirectional_last_seen_ms','bidirectional_duration_ms', 'src2dst_first_seen_ms','src2dst_last_seen_ms', 'src2dst_duration_ms','dst2src_first_seen_ms', 'dst2src_last_seen_ms','dst2src_duration_ms', 'bidirectional_min_piat_ms','bidirectional_mean_piat_ms', 'bidirectional_stddev_piat_ms','bidirectional_max_piat_ms', 'src2dst_min_piat_ms','src2dst_mean_piat_ms','src2dst_stddev_piat_ms', 'src2dst_max_piat_ms','dst2src_min_piat_ms', 'dst2src_mean_piat_ms', 'dst2src_stddev_piat_ms','dst2src_max_piat_ms','datetime']


df1_raw = pd.read_csv(caminho + 'base_Mod1_rotulada.csv')
df2_raw = pd.read_csv(caminho + 'base_Mod2_rotulada.csv')
df3_raw = pd.read_csv(caminho + 'base_Mod3_rotulada.csv')
C2017 = pd.read_csv(r'C:\IC\experimentoIDS\CIC2017-nfstream-target.csv', sep=None, engine='python')
'''
print("\n=== DIAGNÓSTICO GUSTAVO ===")
for nome_atq, lista_dfs in gustavo_por_ataque.items():
    col_esperada = f"attack-{nome_atq}"
    for i, d in enumerate(lista_dfs):
        colunas_attack = [c for c in d.columns if c.startswith('attack')]
        print(f"{nome_atq} | df #{i} | esperava '{col_esperada}' | colunas 'attack*' encontradas: {colunas_attack}")


print("\n=== VERIFICAÇÃO DAS BASES ===")
print("Base 1:"); print(df1_raw['label'].value_counts())
print("Base 2:"); print(df2_raw['label'].value_counts())
print("Base 3:"); print(df3_raw['label'].value_counts())
'''

lista_ataques = ["DoS-Slowhttptest", "DoS-GoldenEye", "DoS-Hulk", "DoS-slowloris", "FTP-Patator", "SSH-Patator", "NMAP-Completo"]


def preparar_ataque_gustavo(lista_dfs, col_target):
    partes = [d for d in lista_dfs if not d.empty]
    if not partes or col_target is None:
        return pd.DataFrame()
    df = pd.concat(partes, ignore_index=True)
    if col_target not in df.columns:
        return pd.DataFrame()
    df = df.rename(columns={col_target: 'target'})
    df['target'] = df['target'].astype(int)
    df = df.select_dtypes(include=['number'])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def preparar_ataque(df_raw, nome_atq):
    #Filtra benigno + ataque, cria target, limpa e retorna as colunas numéricas
    temp = df_raw[(df_raw['label'] == 'benign') | (df_raw['label'] == nome_atq)].copy()
    if temp.empty or len(temp['label'].unique()) < 2:
        return pd.DataFrame()
    temp['target'] = (temp['label'] == nome_atq).astype(int)
    # Seleciona numéricas
    temp = temp.select_dtypes(include=['number'])
    temp = temp.drop(columns=colunas_para_remover, errors='ignore')
    temp = temp.replace([np.inf, -np.inf], np.nan).dropna()
    return temp

def preparar_propensity(df_ataque_processado, origem):
    """
    Recebe um df processado por preparar_ataque (que tem target=1 para
    linhas de ataque e target=0 para benigno) e devolve só as linhas de
    ataque, com o target substituído pela origem da base.
    origem: 0 = CIC, 1 = mnha base
    """
    apenas_ataque = df_ataque_processado[df_ataque_processado['target'] == 1].copy()
    apenas_ataque['target'] = origem
    return apenas_ataque

def balancear_undersampling(df):
    #Reduz a classe majoritária (benigno) até igualar a minoritária (ataque) -> 50/50.
    X, y = df.drop(columns=['target']), df['target']
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    df_res = X_res.copy()
    df_res['target'] = y_res
    return df_res

def balancear_smote(df, n_benignos=200_000):
    #aplica SMOTE nos ataques até balancear 50/50.
    benignos = df[df['target'] == 0]
    ataques  = df[df['target'] == 1]
    if len(benignos) > n_benignos:
        benignos = benignos.sample(n=n_benignos, random_state=42)
    df_reduzido = pd.concat([benignos, ataques])
    X, y = df_reduzido.drop(columns=['target']), df_reduzido['target']
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    df_res = X_res.copy()
    df_res['target'] = y_res
    return df_res

def alinhar_colunas(df_tr, df_ts):
    #Garante que treino e teste tenham as mesmas colunas
    cols_tr = set(df_tr.columns) - {'target'}
    cols_ts = set(df_ts.columns) - {'target'}
    cols_comuns = sorted(cols_tr & cols_ts)
    return df_tr[cols_comuns + ['target']], df_ts[cols_comuns + ['target']]

def treinar_e_avaliar(df_tr, df_ts, nome_cenario):
    #Treina RF e retorna dict com métricas
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
with pd.ExcelWriter('Experimento5.xlsx', engine='xlsxwriter') as writer:
    for ataque in lista_ataques:
        print(f"\nProcessando: {ataque}")

        f1 = preparar_ataque(df1_raw, ataque)
        f2 = preparar_ataque(df2_raw, ataque)
        f3 = preparar_ataque(df3_raw, ataque)
        col_target, lista_dfs_gustavo = gustavo_col_map.get(ataque, (None, [])) 
        fgustavo = preparar_ataque_gustavo(lista_dfs_gustavo, col_target)
        fcic = preparar_ataque(C2017, ataque)

        if f1.empty and f2.empty and f3.empty:
            print(f"   [AVISO] Ataque '{ataque}' não encontrado em nenhuma base local. Pulando.")
            continue

        df_all = pd.concat([d for d in [f1, f2, f3] if not d.empty])

        cenarios_definidos = []

        # -- Cenário BASE (70/30 das bases locais) --
        if not df_all.empty and len(df_all['target'].unique()) > 1:
            try:
                df_tr_base, df_ts_base = train_test_split(
                    df_all, test_size=0.3, random_state=42, stratify=df_all['target']
                )
                cenarios_definidos.append((df_tr_base, df_ts_base, "base"))
            except Exception as e:
                print(f"   [ERRO] Falha ao criar split para o cenário base: {e}")

        # -- Cenários de Validação Cruzada Local (C1 a C6) --
        # (treino, teste, nome)
        validacao_local = [
            ([f1, f2], [f3],      "c1"),
            ([f1, f3], [f2],      "c2"),
            ([f2, f3], [f1],      "c3"),
            ([f1],     [f2, f3],  "c4"),
            ([f2],     [f1, f3],  "c5"),
            ([f3],     [f1, f2],  "c6"),
        ]
        for tr_list, ts_list, nome_c in validacao_local:
            v_tr = pd.concat([d for d in tr_list if not d.empty])
            v_ts = pd.concat([d for d in ts_list if not d.empty])
            if not v_tr.empty and not v_ts.empty:
                cenarios_definidos.append((v_tr, v_ts, nome_c))

        # --Cenários com o CIC2017 --
        if not fcic.empty and len(fcic['target'].unique()) > 1:
            # c7: Treina CIC -> Testa Local
            cenarios_definidos.append((fcic, df_all, "c7"))
            # c8: Treina Local -> Testa CIC
            cenarios_definidos.append((df_all, fcic, "c8"))
            
            # c9 e c10: Fluxo normal isolado do CIC
            cic_normal = fcic[fcic['target'] == 0].copy()
            cic_ataque = fcic[fcic['target'] == 1].copy()
            local_ataques = df_all[df_all['target'] == 1].copy()

            if not cic_normal.empty and not cic_ataque.empty and not local_ataques.empty:
                base1 = pd.concat([cic_normal, cic_ataque], axis=0)  # CIC Puro
                base2 = pd.concat([cic_normal, local_ataques], axis=0) # CIC Normal + Ataque Local
                
                cenarios_definidos.append((base1, base2, "c9"))
                cenarios_definidos.append((base2, base1, "c10"))
        else:
            print(f"   [AVISO] Ataque '{ataque}' ausente/insuficiente no CIC2017. Pulando C7 a C10.")

        # -- Cenários com as Bases do Gustavo (C11 e C12) --
        if not fgustavo.empty and len(fgustavo['target'].unique()) > 1:
            cenarios_definidos.append((fgustavo, df_all, "c11"))
            cenarios_definidos.append((df_all, fgustavo, "c12"))
        else:
            print(f"   [AVISO] Ataque '{ataque}' ausente/insuficiente no Gustavo. Pulando C11 e C12.")

        #-- Cenário de Propensity Score (C13) --
        if not fcic.empty and len(fcic['target'].unique()) > 1 and not df_all.empty:
            cic_prop = preparar_propensity(fcic, origem=0)
            base_prop = preparar_propensity(df_all, origem=1)

            if not cic_prop.empty and not base_prop.empty:
                cic_prop_al, base_prop_al = alinhar_colunas(cic_prop, base_prop)
                df_prop = pd.concat([cic_prop_al, base_prop_al], ignore_index=True)

                if len(df_prop['target'].unique()) > 1:
                    try:
                        df_tr_prop, df_ts_prop = train_test_split(
                            df_prop, test_size=0.3, random_state=42, stratify=df_prop['target']
                        )
                        cenarios_definidos.append((df_tr_prop, df_ts_prop, "c13"))
                    except Exception as e:
                        print(f"   [ERRO] Falha ao criar split para Propensity (C13): {e}")
            else:
                print(f"   [AVISO] Dados de Propensity insuficientes para C13.")
        else:
            print(f"   [AVISO] Pulando C13 por falta de dados (CIC ou Local vazios).")

        # -- Cenários de Balanceamento (C14 e C15) --
        if not fcic.empty and len(fcic['target'].unique()) > 1:
            # c14: Undersampling
            try:
                fcic_under = balancear_undersampling(fcic)
                cenarios_definidos.append((fcic_under, df_all, "c14"))
            except Exception as e:
                print(f"   [ERRO] Falha ao gerar Undersampling (C14) para {ataque}: {e}")

            # c15: SMOTE
            try:
                fcic_smote = balancear_smote(fcic)
                cenarios_definidos.append((fcic_smote, df_all, "c15"))
            except Exception as e:
                print(f"   [ERRO] Falha ao gerar SMOTE (C15) para {ataque}: {e}")


        # -- EXECUÇÃO SEQUENCIAL DE TODOS OS CENÁRIOS GERADOS --
        resultados = []
        for df_treino, df_teste, nome_c in cenarios_definidos:
            if df_treino.empty or df_teste.empty:
                print(f"   [AVISO] Cenário {nome_c} possui bases de treino ou teste vazias. Ignorando.")
                continue
            
            res = treinar_e_avaliar(df_treino, df_teste, nome_c)
            if res:
                resultados.append(res)

        # Salva na planilha mantendo a ordem exata da lista resultados
        if resultados:
            pd.DataFrame(resultados).to_excel(writer, sheet_name=ataque[:31], index=False)
            print(f"   [SUCESSO] Aba '{ataque}' gravada com {len(resultados)} cenários ordenados.")
        else:
            print(f"   [AVISO] Nenhum resultado gerado para '{ataque}'.")

print("\nFim do experimento!")

