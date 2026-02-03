import pandas as pd
import re


def parser_lisa_sheet(lisa_texte_brut):
    """
    Parse le texte brut d'une LISA Sheet et extrait les informations
    """
    
    data = {
        'identifiant': '',
        'rang': '',
        'intitule': '',
        'description': '',
        'rubrique': '',
        'item_parent': '',
        'contenu': ''
    }
    
    # Extraire l'identifiant
    match = re.search(r'\|Identifiant=([^\n|]+)', lisa_texte_brut)
    if match:
        data['identifiant'] = match.group(1).strip()
    
    # Extraire Item_parent
    match = re.search(r'\|Item_parent=([^\n|]*?)(?:\|Item_parent_short|$)', lisa_texte_brut, re.DOTALL)
    if match:
        data['item_parent'] = match.group(1).strip()
    
    # Extraire le rang
    match = re.search(r'\|Rang=([^\n|]+)', lisa_texte_brut)
    if match:
        data['rang'] = match.group(1).strip()
    
    # Extraire l'intitulé
    match = re.search(r'\|Intitulé=([^\n|]+)', lisa_texte_brut)
    if match:
        data['intitule'] = match.group(1).strip()
    
    # Extraire la description
    match = re.search(r'\|Description=([^\n|]+)', lisa_texte_brut)
    if match:
        data['description'] = match.group(1).strip()
    
    # Extraire la rubrique
    match = re.search(r'\|Rubrique=([^\n|]+)', lisa_texte_brut)
    if match:
        data['rubrique'] = match.group(1).strip()
    
    # Extraire le contenu texte après }}
    match = re.search(r'\}\}(.*)', lisa_texte_brut, re.DOTALL)
    if match:
        data['contenu'] = match.group(1).strip()
    
    return data


def evaluation_to_pass_warn(score,threshold):
    """
    Convertit un score numérique ou booléen en PASS ou WARN
    """
    if pd.isna(score):
        return "WARN"
    
    try:
        val = float(score)
        return "PASS" if val >= threshold else "WARN"
    except:
        if isinstance(score, str):
            return "PASS" if score.lower() in ['true', 'yes', 'pass', '1', "medium", 'low'] else "WARN"
        return "PASS" if score else "WARN"


def construire_params_depuis_csv(df,model):
    """
    Construit la liste de paramètres à partir d'un fichier CSV
    
    Colonnes attendues:
    - id: Source material (LISA Sheet ###...)
    - question: Question MCQ
    - option_a, option_b, option_c, option_d: Options
    - correct_option: Option correcte (A, B, C ou D)
    - content_raw: Contenu brut de la LISA Sheet
    - originality: Score (0-1)
    - readability: Score
    - starts_with_negation: True/False
    - is_question: True/False
    - relevance: Score (0-1)
    - ambiguity: Score (0-1)
    - disclosure: Score (0-1)
    - difficulty: Score/Texte
    """
    
    # Lire le CSV    
    params_list = []
    
    for idx, row in df.iterrows():
        # Construire les checks Section A
        section_a_checks = [
            #("Valid JSON schema", "PASS", "required", "Parsed and validated"),
            ("Question mark present", 
             evaluation_to_pass_warn(row.get('is_question'), True), 
             "required", 
             "Question format validated" if evaluation_to_pass_warn(row.get('is_question'), True) == "PASS" else "Not a question"),
            ("No leading negation", 
             "WARN" if row.get('starts_with_negation') == True or str(row.get('starts_with_negation')).lower() == 'true' else "PASS", 
             "required", 
             "Negation detected" if row.get('starts_with_negation') == True or str(row.get('starts_with_negation')).lower() == 'true' else "No negation"),
            ("Originality (integral)", 
             evaluation_to_pass_warn(row.get('originality'),0.75), 
             "≥ 0.75", 
             f"Score: {row.get('originality', 'N/A')}"),
            ("Readability (FK grade)", 
             evaluation_to_pass_warn(row.get('readability'),12), 
             "≥ 12", 
             f"Score: {row.get('readability')}")
        ]
        
        # Construire les checks Section B
        difficulty = int(row.get('difficulty'))
        mapping = {
                1: "low",
                2: "low",
                3: "medium",
                4: "high",
                5: "high"
            }

        section_b_checks = [
            ("Disclosure (answer leakage)", 
             evaluation_to_pass_warn(row.get('disclosure'),True), 
             "True/False", 
             f"Score: {row.get('disclosure', 'N/A')}"),
            ("Relevance to material", 
             evaluation_to_pass_warn(row.get('relevance'),0.8), # To define
             "0-1", 
             f"Score: {row.get('relevance', 'N/A')}"),
            ("Distractor plausibility", 
             evaluation_to_pass_warn(row.get('distractor_quality'),True), # To define
             "True/False", 
             f"Score: {row.get('distractor_quality', 'N/A')}"),
            ("Difficulty appropriateness", 
             evaluation_to_pass_warn(difficulty,"medium"), # To define
             "low/med/high", 
             f"Judge: {mapping[difficulty]}") # Une ligne pour l'ambiguité
        ]
        
        # Construire le dictionnaire de paramètres
        params = {
            "item_id": f"MCQ-{idx+1:06d}",
            "source_material": str(row.get('id', 'LISA Sheet')),
            "generator_info": model,
            "output_format": "CSV",
            "mcq_question": str(row.get('question', '')),
            "options": {
                "A": str(row.get('option_a', '')),
                "B": str(row.get('option_b', '')),
                "C": str(row.get('option_c', '')),
                "D": str(row.get('option_d', ''))
            },
            "correct_option": str(row.get('correct_option', 'A')),
            "section_a_checks": section_a_checks,
            "section_b_checks": section_b_checks,
            "decision_policy": "Accept if all hard constraints pass and no critical AI-judge dimension fails.",
            "final_decision": "ACCEPT" if all(check[1] == "PASS" for check in section_a_checks + section_b_checks) else "REVISE",
            "audit_trail": "Judge model: Automated, consistency: evaluated from CSV metrics.",
            "lisa_texte_brut": str(row.get('content_raw', ''))
        }
        
        params_list.append(params)
    
    return params_list


def generer_carte_mcq(card_data, index):
    """
    Génère le HTML pour une seule carte MCQ
    """
    
    # Parser LISA
    lisa_data = parser_lisa_sheet(card_data['lisa_texte_brut'])
    
    # Générer Section A HTML
    section_a_html = ""
    for dim, result, threshold, notes in card_data['section_a_checks']:
        classe_result = "pass" if result == "PASS" else "warn"
        section_a_html += f"""
        <tr>
            <td>{dim}</td>
            <td class="result {classe_result}">{result}</td>
            <td>{threshold}</td>
            <td>{notes}</td>
        </tr>
        """
    
    # Générer Section B HTML
    section_b_html = ""
    for dim, result, score, notes in card_data['section_b_checks']:
        classe_result = "pass" if result == "PASS" else "warn"
        section_b_html += f"""
        <tr>
            <td>{dim}</td>
            <td class="result {classe_result}">{result}</td>
            <td>{score}</td>
            <td>{notes}</td>
        </tr>
        """
    
    # Générer les options
    options_html = ""
    for lettre in ["A", "B", "C", "D"]:
        option_text = card_data['options'].get(lettre, '')
        if option_text and str(option_text).strip():
            options_html += f"<p><strong>{lettre})</strong> {option_text}</p>\n"
    
    # Convertir le contenu LISA
    lisa_contenu_html = lisa_data['contenu'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br/>')
    
    html = f"""
    <div class="slide" id="slide-{index}">
        <div class="main-container">
            <!-- Colonne MCQ Card -->
            <div class="column">
                <div class="card">
                    <div class="header">MCQ Acceptance Card (example template)</div>
                    
                    <div class="header-info">
                        <p><strong>Item ID:</strong> {card_data['item_id']}</p>
                        <p><strong>Source Material:</strong> {card_data['source_material']}</p>
                        <p><strong>Generator:</strong> {card_data['generator_info']}</p>
                        <p><strong>Output Format:</strong> {card_data['output_format']}</p>
                    </div>
                    
                    <div class="mcq-section">
                        <h3>MCQ:</h3>
                        <p>{card_data['mcq_question']}</p>
                        <div class="options">
                            {options_html}
                        </div>
                        <p class="correct-option"><strong>Correct option:</strong> {card_data['correct_option']}</p>
                    </div>
                    
                    <div class="section-title">Section A — Deterministic Checks</div>
                    <table>
                        <thead>
                            <tr>
                                <th>Dimension</th>
                                <th>Result</th>
                                <th>Threshold</th>
                                <th>Notes</th>
                            </tr>
                        </thead>
                        <tbody>
                            {section_a_html}
                        </tbody>
                    </table>
                    
                    <div class="section-title">Section B — Judgment-Based Checks</div>
                    <table>
                        <thead>
                            <tr>
                                <th>Dimension</th>
                                <th>Result</th>
                                <th>Score</th>
                                <th>Notes</th>
                            </tr>
                        </thead>
                        <tbody>
                            {section_b_html}
                        </tbody>
                    </table>
                    
                    <div class="decision-section">
                        <p><strong>Decision Policy</strong></p>
                        <p>{card_data['decision_policy']}</p>
                        <p style="margin-top: 8px;"><strong>Final decision:</strong> {card_data['final_decision']}</p>
                    </div>
                    
                    <div class="audit-trail">
                        <h4>Audit Trail</h4>
                        <p>{card_data['audit_trail']}</p>
                    </div>
                </div>
            </div>
            
            <!-- Colonne LISA Sheet -->
            <div class="column">
                <div class="card">
                    <div class="lisa-header">LISA Sheet - Objectif de Connaissance</div>
                    
                    <div class="lisa-metadata">
                        <p><strong>Identifiant:</strong> {lisa_data['identifiant']}</p>
                        <p><strong>Rang:</strong> {lisa_data['rang']}</p>
                        <p><strong>Rubrique:</strong> {lisa_data['rubrique']}</p>
                    </div>
                    
                    <div class="lisa-section">
                        <div class="lisa-section-title">Intitulé</div>
                        <div class="lisa-section-content">{lisa_data['intitule']}</div>
                    </div>
                    
                    <div class="lisa-item-parent">
                        <div class="lisa-item-parent-label">Item Parent:</div>
                        <div>{lisa_data['item_parent']}</div>
                    </div>
                    
                    <div class="lisa-section">
                        <div class="lisa-section-title">Description</div>
                        <div class="lisa-section-content">{lisa_data['description']}</div>
                    </div>
                    
                    <div class="lisa-section">
                        <div class="lisa-section-title">Contenu</div>
                        <div class="lisa-section-content">{lisa_contenu_html}</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    
    return html


def generer_mcq_multi_cartes(params_list, filename="mcq_multi_cartes.html"):
    """
    Génère une page avec plusieurs cartes MCQ + LISA Sheets avec défilement
    """
    
    # Générer les cartes
    cartes_html = ""
    for i, params in enumerate(params_list):
        cartes_html += generer_carte_mcq(params, i)
    
    nombre_cartes = len(params_list)
    
    # HTML complet
    html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCQ Acceptance Cards</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            overflow-x: hidden;
        }}
        
        .container {{
            width: 100%;
            height: 100vh;
            position: relative;
            overflow: hidden;
        }}
        
        .slides-wrapper {{
            display: flex;
            height: 100%;
            transition: transform 0.6s ease-in-out;
        }}
        
        .slide {{
            flex: 0 0 100%;
            width: 100%;
            height: 100%;
            overflow-y: auto;
            padding: 20px;
            scroll-behavior: smooth;
        }}
        
        .main-container {{
            max-width: 1600px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            height: fit-content;
            min-height: 100%;
        }}
        
        .column {{
            display: flex;
            flex-direction: column;
        }}
        
        .card {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        
        .header {{
            background-color: #999;
            color: white;
            padding: 12px 15px;
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 15px;
            border-radius: 4px;
        }}
        
        .header-info {{
            margin-bottom: 15px;
            font-size: 13px;
        }}
        
        .header-info p {{
            margin: 4px 0;
            line-height: 1.5;
        }}
        
        .header-info strong {{
            font-weight: 600;
        }}
        
        .mcq-section {{
            margin: 15px 0;
            padding: 12px;
            background-color: #f9f9f9;
            border-left: 4px solid #333;
        }}
        
        .mcq-section h3 {{
            margin-bottom: 8px;
            font-size: 14px;
        }}
        
        .mcq-section p {{
            margin: 6px 0;
            line-height: 1.5;
            font-size: 13px;
        }}
        
        .options {{
            margin: 10px 0;
            padding-left: 15px;
        }}
        
        .options p {{
            margin: 5px 0;
        }}
        
        .correct-option {{
            font-weight: bold;
            color: #333;
            margin-top: 8px;
        }}
        
        .section-title {{
            font-size: 14px;
            font-weight: bold;
            margin-top: 15px;
            margin-bottom: 10px;
            padding-bottom: 6px;
            border-bottom: 2px solid #333;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 12px;
        }}
        
        table th {{
            background-color: #f0f0f0;
            padding: 8px;
            text-align: left;
            font-weight: 600;
            border-bottom: 1px solid #999;
        }}
        
        table td {{
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }}
        
        table tr:hover {{
            background-color: #f9f9f9;
        }}
        
        .result {{
            font-weight: bold;
            text-align: center;
        }}
        
        .result.pass {{
            color: #28a745;
        }}
        
        .result.warn {{
            color: #ff8c00;
        }}
        
        .decision-section {{
            margin: 15px 0;
            padding: 12px;
            background-color: #fff3cd;
            border-left: 4px solid #ff8c00;
            border-radius: 4px;
        }}
        
        .decision-section p {{
            margin: 4px 0;
            font-size: 13px;
        }}
        
        .decision-section strong {{
            font-weight: bold;
        }}
        
        .audit-trail {{
            margin-top: 15px;
            padding: 12px;
            background-color: #f0f0f0;
            border-radius: 4px;
            font-size: 12px;
            line-height: 1.5;
        }}
        
        .audit-trail h4 {{
            margin-bottom: 8px;
            font-size: 13px;
            font-weight: bold;
        }}
        
        /* LISA Sheet Styles */
        .lisa-header {{
            background-color: #333;
            color: white;
            padding: 12px 15px;
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 15px;
            border-radius: 4px;
        }}
        
        .lisa-metadata {{
            margin-bottom: 15px;
            font-size: 13px;
            line-height: 1.6;
        }}
        
        .lisa-metadata p {{
            margin: 5px 0;
        }}
        
        .lisa-metadata strong {{
            font-weight: 600;
        }}
        
        .lisa-item-parent {{
            background-color: #e8f4f8;
            padding: 12px;
            border-left: 4px solid #0077be;
            border-radius: 4px;
            margin: 12px 0;
            font-size: 13px;
            line-height: 1.6;
        }}
        
        .lisa-item-parent-label {{
            font-weight: bold;
            margin-bottom: 6px;
        }}
        
        .lisa-section {{
            margin: 15px 0;
            padding: 12px;
            background-color: #f9f9f9;
            border-left: 4px solid #666;
            border-radius: 4px;
        }}
        
        .lisa-section-title {{
            font-weight: bold;
            margin-bottom: 8px;
            font-size: 13px;
        }}
        
        .lisa-section-content {{
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
        }}
        
        /* Navigation */
        .navigation {{
            position: fixed;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 15px;
            z-index: 100;
            background: white;
            padding: 15px 25px;
            border-radius: 50px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        
        .nav-button {{
            background-color: #333;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: background-color 0.3s;
        }}
        
        .nav-button:hover {{
            background-color: #555;
        }}
        
        .nav-button:disabled {{
            background-color: #ccc;
            cursor: not-allowed;
        }}
        
        .slide-counter {{
            display: flex;
            align-items: center;
            gap: 10px;
            color: #333;
            font-weight: 600;
        }}
        
        .slide-counter input {{
            width: 50px;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            text-align: center;
            font-size: 14px;
        }}
        
        .progress-bar {{
            position: fixed;
            top: 0;
            left: 0;
            height: 3px;
            background-color: #007bff;
            transition: width 0.3s;
            z-index: 101;
        }}
        
        /* Responsive */
        @media (max-width: 1400px) {{
            .main-container {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="progress-bar" id="progressBar"></div>
    
    <div class="container">
        <div class="slides-wrapper" id="slidesWrapper">
            {cartes_html}
        </div>
    </div>
    
    <div class="navigation">
        <button class="nav-button" id="prevBtn" onclick="previousSlide()">← Précédent</button>
        <div class="slide-counter">
            <span>Carte</span>
            <input type="number" id="slideInput" min="1" max="{nombre_cartes}" value="1" onchange="goToSlide(this.value)">
            <span>/ {nombre_cartes}</span>
        </div>
        <button class="nav-button" id="nextBtn" onclick="nextSlide()">Suivant →</button>
    </div>
    
    <script>
        let currentSlide = 0;
        const totalSlides = {nombre_cartes};
        const slidesWrapper = document.getElementById('slidesWrapper');
        const slideInput = document.getElementById('slideInput');
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        const progressBar = document.getElementById('progressBar');
        
        function showSlide(n) {{
            if (n >= totalSlides) {{
                currentSlide = totalSlides - 1;
            }}
            if (n < 0) {{
                currentSlide = 0;
            }}
            
            const offset = -currentSlide * 100;
            slidesWrapper.style.transform = `translateX(${{offset}}%)`;
            
            slideInput.value = currentSlide + 1;
            prevBtn.disabled = currentSlide === 0;
            nextBtn.disabled = currentSlide === totalSlides - 1;
            
            const progress = ((currentSlide + 1) / totalSlides) * 100;
            progressBar.style.width = progress + '%';
        }}
        
        function nextSlide() {{
            currentSlide++;
            showSlide(currentSlide);
        }}
        
        function previousSlide() {{
            currentSlide--;
            showSlide(currentSlide);
        }}
        
        function goToSlide(n) {{
            currentSlide = parseInt(n) - 1;
            showSlide(currentSlide);
        }}
        
        // Contrôles au clavier
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'ArrowRight') {{
                nextSlide();
            }} else if (event.key === 'ArrowLeft') {{
                previousSlide();
            }}
        }});
        
        // Initialiser
        showSlide(currentSlide);
    </script>
</body>
</html>
"""
    
    # Sauvegarder le fichier
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✓ Fichier HTML généré : {filename}")
    print(f"✓ Nombre de cartes : {nombre_cartes}")
    return filename


# Utilisation
if __name__ == "__main__":
    # Lire le CSV
    csv_file = "df_eval.csv"  # À remplacer par ton chemin réel
    df = pd.read_csv("...")
    print(f"Lecture du fichier CSV: {csv_file}")
    params_list = construire_params_depuis_csv(df)
    
    print(f"✓ {len(params_list)} cartes construites")
    
    # Générer le fichier HTML
    generer_mcq_multi_cartes(params_list, filename="output_mcq.html")
    
    print("✓ Génération terminée!")