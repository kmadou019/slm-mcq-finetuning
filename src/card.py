import pandas as pd
import re
import json


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


def evaluation_to_pass_warn(score, threshold):
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


def construire_params_depuis_csv(df, model):
    """
    Construit la liste de paramètres à partir d'un DataFrame
    """
    
    params_list = []
    
    for idx, row in df.iterrows():
        # Construire les checks Section A
        section_a_checks = [
            ("Question mark present", 
             evaluation_to_pass_warn(row.get('is_question'), True), 
             "required", 
             "Question format validated" if evaluation_to_pass_warn(row.get('is_question'), True) == "PASS" else "Not a question"),
            ("No leading negation", 
             "WARN" if row.get('starts_with_negation') == True or str(row.get('starts_with_negation')).lower() == 'true' else "PASS", 
             "required", 
             "Negation detected" if row.get('starts_with_negation') == True or str(row.get('starts_with_negation')).lower() == 'true' else "No negation"),
            ("Originality (integral)", 
             evaluation_to_pass_warn(row.get('originality'), 0.75), 
             "≥ 0.75", 
             f"Score: {row.get('originality', 'N/A')}"),
            ("Readability (FK grade)", 
             evaluation_to_pass_warn(row.get('readability'), 12), 
             "≥ 12", 
             f"Score: {row.get('readability')}")
        ]
        
        # Construire les checks Section B
        difficulty = int(row.get('difficulty', 3))
        mapping = {
            1: "low",
            2: "low",
            3: "medium",
            4: "high",
            5: "high"
        }

        section_b_checks = [
            ("Disclosure (answer leakage)", 
             evaluation_to_pass_warn(row.get('disclosure'), True), 
             "True/False", 
             f"Score: {row.get('disclosure', 'N/A')}"),
            ("Relevance to material", 
             evaluation_to_pass_warn(row.get('relevance'), 0.8),
             "0-1", 
             f"Score: {row.get('relevance', 'N/A')}"),
            ("Distractor plausibility", 
             evaluation_to_pass_warn(row.get('distractor_quality'), True),
             "True/False", 
             f"Score: {row.get('distractor_quality', 'N/A')}"),
            ("Difficulty appropriateness", 
             evaluation_to_pass_warn(difficulty, "medium"),
             "low/med/high", 
             f"Judge: {mapping[difficulty]}")
        ]
        
        # Construire le dictionnaire de paramètres
        params = {
            "item_id": f"MCQ-{idx+1:06d}",
            "source_material": str(row.get('id', 'LISA Sheet')),
            "generator_info": model,
            "output_format": "JSON",
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
    Génère le HTML pour une seule carte MCQ avec champ feedback
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
            <td class="validation-cell">
                <input type="checkbox" class="validation-checkbox" data-type="section-a" data-index="{index}" data-dim="{dim}">
            </td>
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
            <td class="validation-cell">
                <input type="checkbox" class="validation-checkbox" data-type="section-b" data-index="{index}" data-dim="{dim}">
            </td>
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
                                <th>Validation</th>
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
                                <th>Validation</th>
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
                    
                    <div class="decision-buttons">
                        <button class="btn-accept" onclick="acceptCard({index})">✓ Accepter</button>
                        <button class="btn-reject" onclick="rejectCard({index})">✗ Refuser</button>
                    </div>
                    
                    <div class="feedback-section">
                        <label for="feedback-{index}"><strong>Feedback (optionnel):</strong></label>
                        <textarea id="feedback-{index}" class="feedback-textarea" placeholder="Entrez votre feedback ici..." onchange="saveFeedback({index})" data-index="{index}"></textarea>
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
    Génère une page avec plusieurs cartes MCQ + LISA Sheets avec défilement, validation et feedback
    """
    
    # Générer les cartes
    cartes_html = ""
    for i, params in enumerate(params_list):
        cartes_html += generer_carte_mcq(params, i)
    
    nombre_cartes = len(params_list)
    
    # Préparer les données pour le JavaScript (JSON des cartes)
    cartes_json = json.dumps([{
        "index": i,
        "source_material": params["source_material"],
        "mcq_question": params["mcq_question"],
        "options": params["options"],
        "correct_option": params["correct_option"]
    } for i, params in enumerate(params_list)])
    
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
        
        .validation-cell {{
            text-align: center;
            padding: 8px;
        }}
        
        .validation-checkbox {{
            width: 20px;
            height: 20px;
            cursor: pointer;
            accent-color: #28a745;
        }}
        
        .validation-checkbox:hover {{
            transform: scale(1.1);
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
        
        .decision-buttons {{
            display: flex;
            gap: 10px;
            margin-top: 20px;
            justify-content: center;
        }}
        
        .btn-accept {{
            background-color: #28a745;
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: background-color 0.3s;
            flex: 1;
        }}
        
        .btn-accept:hover {{
            background-color: #218838;
        }}
        
        .btn-reject {{
            background-color: #dc3545;
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: background-color 0.3s;
            flex: 1;
        }}
        
        .btn-reject:hover {{
            background-color: #c82333;
        }}
        
        .btn-accept.selected {{
            border: 3px solid #0056b3;
        }}
        
        .btn-reject.selected {{
            border: 3px solid #0056b3;
        }}
        
        .feedback-section {{
            margin-top: 20px;
            padding: 15px;
            background-color: #e7f3ff;
            border-left: 4px solid #0077be;
            border-radius: 4px;
        }}
        
        .feedback-section label {{
            display: block;
            margin-bottom: 8px;
            font-size: 13px;
            color: #333;
        }}
        
        .feedback-textarea {{
            width: 100%;
            min-height: 100px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 13px;
            resize: vertical;
            border: 2px solid #0077be;
        }}
        
        .feedback-textarea:focus {{
            outline: none;
            border-color: #005a9f;
            box-shadow: 0 0 5px rgba(0, 119, 182, 0.3);
        }}
        
        .feedback-textarea::placeholder {{
            color: #999;
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
        <button class="nav-button" id="exportBtn" onclick="exportValidations()" style="background-color: #007bff;">📥 Exporter</button>
    </div>
    
    <script>
        // Clé unique pour cette génération (basée sur le nombre de cartes et les données)
        const generationKey = 'mcq_validations_' + {nombre_cartes} + '_' + JSON.stringify({cartes_json}).substring(0, 50).replace(/[^a-zA-Z0-9]/g, '');
        
        // Stockage des validations, des décisions et des feedbacks
        let validations = {{}};
        let cardDecisions = {{}};
        let cardFeedbacks = {{}};
        const cardData = {cartes_json};
        
        let currentSlide = 0;
        const totalSlides = {nombre_cartes};
        const slidesWrapper = document.getElementById('slidesWrapper');
        const slideInput = document.getElementById('slideInput');
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        const progressBar = document.getElementById('progressBar');
        
        // Initialiser les décisions et feedbacks
        for (let i = 0; i < totalSlides; i++) {{
            cardDecisions[i] = null;
            cardFeedbacks[i] = '';
        }}
        
        // Charger les validations depuis localStorage si la clé correspond
        function loadValidations() {{
            try {{
                const stored = localStorage.getItem(generationKey);
                if (stored) {{
                    const data = JSON.parse(stored);
                    validations = data.validations || {{}};
                    cardDecisions = data.decisions || {{}};
                    cardFeedbacks = data.feedbacks || {{}};
                    
                    // Restaurer les checkboxes
                    document.querySelectorAll('.validation-checkbox').forEach(checkbox => {{
                        const key = `${{checkbox.dataset.type}}_${{checkbox.dataset.index}}_${{checkbox.dataset.dim}}`;
                        if (validations[key] !== undefined) {{
                            checkbox.checked = validations[key];
                        }}
                    }});
                    
                    // Restaurer les feedbacks
                    for (let i = 0; i < totalSlides; i++) {{
                        const feedbackTextarea = document.getElementById(`feedback-${{i}}`);
                        if (feedbackTextarea && cardFeedbacks[i]) {{
                            feedbackTextarea.value = cardFeedbacks[i];
                        }}
                    }}
                    
                    // Restaurer les boutons
                    for (let i = 0; i < totalSlides; i++) {{
                        updateButtonStates(i);
                    }}
                    
                    console.log('✓ Validations et feedbacks restaurés depuis localStorage');
                }}
            }} catch (e) {{
                console.error('Erreur lors du chargement des validations:', e);
            }}
        }}
        
        // Sauvegarder les validations dans localStorage
        function saveValidations() {{
            try {{
                const data = {{
                    validations: validations,
                    decisions: cardDecisions,
                    feedbacks: cardFeedbacks,
                    timestamp: new Date().toISOString()
                }};
                localStorage.setItem(generationKey, JSON.stringify(data));
            }} catch (e) {{
                console.error('Erreur lors de la sauvegarde:', e);
            }}
        }}
        
        // Écouter les changements des cases à cocher
        document.querySelectorAll('.validation-checkbox').forEach(checkbox => {{
            checkbox.addEventListener('change', function() {{
                const key = `${{this.dataset.type}}_${{this.dataset.index}}_${{this.dataset.dim}}`;
                validations[key] = this.checked;
                saveValidations();
            }});
        }});
        
        function saveFeedback(index) {{
            const feedbackTextarea = document.getElementById(`feedback-${{index}}`);
            if (feedbackTextarea) {{
                cardFeedbacks[index] = feedbackTextarea.value;
                saveValidations();
            }}
        }}
        
        function acceptCard(index) {{
            cardDecisions[index] = 'ACCEPT';
            updateButtonStates(index);
            saveValidations();
        }}
        
        function rejectCard(index) {{
            cardDecisions[index] = 'REJECT';
            updateButtonStates(index);
            saveValidations();
        }}
        
        function updateButtonStates(index) {{
            const slide = document.getElementById(`slide-${{index}}`);
            if (slide) {{
                const acceptBtn = slide.querySelector('.btn-accept');
                const rejectBtn = slide.querySelector('.btn-reject');
                
                if (acceptBtn && rejectBtn) {{
                    acceptBtn.classList.remove('selected');
                    rejectBtn.classList.remove('selected');
                    
                    if (cardDecisions[index] === 'ACCEPT') {{
                        acceptBtn.classList.add('selected');
                    }} else if (cardDecisions[index] === 'REJECT') {{
                        rejectBtn.classList.add('selected');
                    }}
                }}
            }}
        }}
        
        function exportValidations() {{
            const exportData = [];
            
            for (let i = 0; i < totalSlides; i++) {{
                // Collecter les validations pour cette carte
                const cardValidations = {{}};
                for (let key in validations) {{
                    if (key.includes(`_${{i}}_`)) {{
                        cardValidations[key] = validations[key];
                    }}
                }}
                
                const cardInfo = {{
                    index: i,
                    id_lisa_sheet: cardData[i].source_material,
                    mcq: {{
                        question: cardData[i].mcq_question,
                        options: cardData[i].options,
                        correct_option: cardData[i].correct_option
                    }},
                    validated_fields: cardValidations,
                    human_decision: cardDecisions[i],
                    human_feedback: cardFeedbacks[i] || null,
                    timestamp: new Date().toISOString()
                }};
                
                exportData.push(cardInfo);
            }}
            
            const json = JSON.stringify(exportData, null, 2);
            const blob = new Blob([json], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'validations_' + new Date().toISOString().slice(0,10) + '.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            alert('✓ Validations et feedbacks exportés en JSON');
        }}
        
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
        loadValidations();
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
    df = pd.read_csv("df_eval.csv")  # À remplacer par ton chemin réel
    print(f"Lecture du fichier CSV")
    params_list = construire_params_depuis_csv(df, "Llama3.1-8B")
    
    print(f"✓ {len(params_list)} cartes construites")
    
    # Générer le fichier HTML
    generer_mcq_multi_cartes(params_list, filename="output_mcq.html")
    
    print("✓ Génération terminée!")