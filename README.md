#  IDS-Random-Forest: Avaliação Cross-Dataset e Análise de Concept Drift

> Repositório direcionado para o armazenamento do algoritmo desenvolvido para a análise de Sistemas de Detecção de Intrusão (IDS) durante o projeto de Iniciação Científica.

### Sobre o Projeto
Este projeto avalia o modelo preditivo baseado no algoritmo **Random Forest** para a detecção de anomalias e tráfego malicioso em redes de computadores. Foram feitas validações cruzadas complexas (*cross-dataset*), testando o modelo contra bases de dados próprias, bases de terceiros (Bases Gustavo) e o renomado dataset público **CICIDS2017**.

O algoritmo automatiza o treinamento, teste e extração de métricas de desempenho abrangendo **15 cenários experimentais** distintos para 7 tipos de ataques.

## Cenários de Avaliação
O script `main.py` (ou nome do seu arquivo) avalia de forma automatizada os seguintes cenários:
* **Cenário Base:** Divisão clássica 70/30 (Treino/Teste) nas bases locais.
* **C1 a C6:** Validação cruzada iterativa entre 3 bases de captura locais.
* **C7 e C8:** Validação *cross-dataset* isolando ataques entre as bases locais e o CICIDS2017.
* **C9 e C10:** Injeção de tráfego de ataque local dentro do fluxo benigno do CICIDS2017.
* **C11 e C12:** Validação cruzada contra bases independentes geradas em outro laboratório (Bases Gustavo).
* **C13 (Propensity Score):** Análise de similaridade e viés de distribuição entre as amostras de ataque das bases.
* **C14 (Undersampling) e C15 (SMOTE):** Avaliação do impacto de técnicas de balanceamento de classes em bases de dados severamente desbalanceadas.

## 🚀 Como Executar

**1. Clone o repositório:**
git clone https://github.com/SEU-USUARIO/IDS-Random-Forest.git

**2. Instale as dependências:**
pip install pandas numpy scikit-learn imbalanced-learn xlsxwriter

**3. Estrutura de Diretórios:**
O script requer que os datasets em formato `.csv` estejam estruturados nos seguintes diretórios locais :
* `C:\IC\experimentoIDS\mods_rotulados\` (Bases Mod1, Mod2, Mod3)
* `C:\IC\basesGustavo\` (Bases independentes)
* `C:\IC\experimentoIDS\` (Base CICIDS2017 - arquivo `CIC2017-nfstream-target.csv`)

**4. Execução:**
Basta executar o script Python. O algoritmo irá processar todos os ataques, treinar as árvores e gerar um relatório completo no arquivo `Experimento5.xlsx` contendo as métricas de AUC, F1-Score, Recall e Precisão.
