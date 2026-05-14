import { Component, EventEmitter, Output, inject, OnDestroy, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { McqService } from '../../services/mcq.service';

type ModalView = 'form' | 'generating' | 'done' | 'error';

export const DEFAULT_PROMPT = `À partir du contenu éducatif suivant, générez exactement une question à choix multiple avec quatre options de réponse (a, b, c, d), dont une seule est correcte.

OBJECTIFS :
- La question doit évaluer la compréhension des idées principales.
- Les distracteurs doivent être plausibles mais incorrects.
- Les options doivent être courtes.
- Fournir une justification pédagogique pour chaque option.
- Fournir un commentaire global pour la question.

CONTRAINTES STRICTES DE SORTIE :
1. La sortie doit être STRICTEMENT un unique objet JSON valide.
2. Interdiction ABSOLUE d'ajouter :
   - des blocs \`\`\`json
   - du texte avant ou après le JSON
   - des explications hors champs JSON
3. Le champ "correct_option" doit contenir EXACTEMENT une lettre minuscule parmi : "a", "b", "c", "d".
4. Utiliser uniquement des doubles quotes : "..."
5. Le JSON doit contenir EXACTEMENT les 11 champs suivants :

{
  "question": "...",
  "question_comment": "...",
  "option_a": "...",
  "option_a_comment": "...",
  "option_b": "...",
  "option_b_comment": "...",
  "option_c": "...",
  "option_c_comment": "...",
  "option_d": "...",
  "option_d_comment": "...",
  "correct_option": "a"
}

RÈGLES POUR LES COMMENTAIRES :
- Chaque commentaire d'option doit expliquer brièvement pourquoi l'option est correcte ou incorrecte.
- Le commentaire global de la question doit expliquer ce que la question évalue ou signaler un piège courant.
- Les commentaires doivent être factuels, concis et pédagogiques.

CONTENU ÉDUCATIF :
{content}

INSTRUCTION FINALE :
Répondez UNIQUEMENT avec un unique objet JSON valide, sans aucun texte en dehors.`;

@Component({
  selector: 'app-generate-from-material-modal',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './generate-from-material-modal.component.html',
  styleUrl: './generate-from-material-modal.component.scss'
})
export class GenerateFromMaterialModalComponent implements OnInit, OnDestroy {
  private readonly mcqService = inject(McqService);
  private readonly cdr = inject(ChangeDetectorRef);

  @Output() close = new EventEmitter<void>();
  @Output() confirm = new EventEmitter<void>();

  // Modèles disponibles pour la génération (chargés dynamiquement depuis le backend)
  generationModels: { label: string, value: string, source: string }[] = [];
  modelsLoading = true;
  modelsError: string | null = null;

  // Champs du formulaire (propriétés simples pour ngModel)
  content = '';
  fileName = '';
  selectedModel = 'hf.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Q8_0';
  promptTemplate = DEFAULT_PROMPT;

  // Prompt editor
  showPromptEditor = false;
  promptState: 'default' | 'enriched' | 'modified' = 'default';
  builtItemRef: string | null = null;
  builtObjectivesCount = 0;
  isBuilding = false;

  // État de la vue
  view: ModalView = 'form';
  errorMessage = '';
  mcqCount = 0;

  ngOnInit(): void {
    this.loadGenerationModels();
  }

  private loadGenerationModels(): void {
    this.modelsLoading = true;
    this.modelsError = null;
    this.mcqService.getGenerationModels().subscribe({
      next: (response) => {
        this.generationModels = response.models;
        this.modelsLoading = false;
        // Auto-select first model if current selection is not available
        if (this.generationModels.length > 0 && !this.generationModels.some(m => m.value === this.selectedModel)) {
          this.selectedModel = this.generationModels[0].value;
        }
        if (response.errors && response.errors.length > 0) {
          console.warn('Model loading warnings:', response.errors);
        }
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Failed to load generation models:', err);
        this.modelsError = 'Impossible de charger les modèles';
        this.modelsLoading = false;
        // Fallback to defaults
        this.generationModels = [
          { label: 'Nemotron 30B (Q8)', value: 'hf.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Q8_0', source: 'default' },
          { label: 'Qwen 3.5 35B', value: 'qwen3.5:35b', source: 'default' },
        ];
      }
    });
  }

  private jobId: string | null = null;
  private pollingIntervalId: any = null;

  get charCount(): number {
    return this.content.length;
  }

  get isContentValid(): boolean {
    return this.content.trim().length >= 100;
  }

  get isPromptModified(): boolean {
    return this.promptState !== 'default';
  }

  ngOnDestroy(): void {
    this.stopPolling();
  }

  togglePromptEditor(): void {
    this.showPromptEditor = !this.showPromptEditor;
  }

  resetPrompt(): void {
    this.promptTemplate = DEFAULT_PROMPT;
    this.promptState = 'default';
    this.builtItemRef = null;
    this.builtObjectivesCount = 0;
  }

  onPromptChange(value: string): void {
    this.promptTemplate = value;
    if (value === DEFAULT_PROMPT) {
      this.promptState = 'default';
      this.builtItemRef = null;
      this.builtObjectivesCount = 0;
    } else {
      this.promptState = 'modified';
    }
  }

  onBuildPrompt(): void {
    this.isBuilding = true;
    this.mcqService.buildPrompt(this.content).subscribe({
      next: (result) => {
        console.log('[buildPrompt] result:', result);
        this.promptTemplate = result.prompt;
        this.builtItemRef = result.item_ref;
        this.builtObjectivesCount = result.objectives_count;
        this.promptState = result.item_ref ? 'enriched' : 'modified';
        this.isBuilding = false;
        this.showPromptEditor = true;
        console.log('[buildPrompt] showPromptEditor:', this.showPromptEditor, 'promptTemplate length:', this.promptTemplate?.length);
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.isBuilding = false;
        this.errorMessage = err?.error?.detail || err?.message || 'Échec de l\'enrichissement LISA.';
        this.cdr.detectChanges();
      }
    });
  }

  onSubmit(): void {
    if (!this.isContentValid) {
      this.errorMessage = 'Le contenu doit contenir au moins 100 caractères.';
      return;
    }
    if (!this.selectedModel) {
      this.errorMessage = 'Veuillez sélectionner un modèle.';
      return;
    }

    this.view = 'generating';
    this.errorMessage = '';

    this.mcqService.startGeneration(this.content, this.selectedModel, this.promptTemplate).subscribe({
      next: (response) => {
        this.jobId = response.job_id;
        this.startPolling();
      },
      error: (err) => {
        this.view = 'error';
        this.errorMessage = err.error?.detail || 'Une erreur est survenue lors du lancement de la génération.';
      }
    });
  }

  private startPolling(): void {
    this.pollingIntervalId = setInterval(() => {
      if (!this.jobId) return;

      this.mcqService.getGenerationStatus(this.jobId).subscribe({
        next: (status) => {
          if (status.status === 'done') {
            this.stopPolling();
            this.mcqCount = status.mcq_count;
            this.view = 'done';
            this.cdr.detectChanges();
          } else if (status.status === 'error') {
            this.stopPolling();
            this.view = 'error';
            this.errorMessage = status.error || 'La génération a échoué.';
            this.cdr.detectChanges();
          }
        },
        error: () => { /* erreurs réseau transitoires ignorées */ }
      });
    }, 5000);
  }

  private stopPolling(): void {
    if (this.pollingIntervalId !== null) {
      clearInterval(this.pollingIntervalId);
      this.pollingIntervalId = null;
    }
  }

  onGoToEvaluation(): void {
    this.confirm.emit();  // le dashboard gère la navigation
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    this.fileName = file.name;
    input.value = '';

    if (file.type === 'application/pdf') {
      this.mcqService.extractPdfText(file).subscribe({
        next: (result) => { this.content = result.text; this.cdr.detectChanges(); },
        error: () => { this.errorMessage = 'Impossible d\'extraire le texte de ce PDF.'; this.cdr.detectChanges(); }
      });
    } else {
      const reader = new FileReader();
      reader.onload = (e) => { this.content = e.target?.result as string; this.cdr.detectChanges(); };
      reader.readAsText(file, 'utf-8');
    }
  }

  onRetry(): void {
    this.stopPolling();
    this.view = 'form';
    this.errorMessage = '';
    this.jobId = null;
    this.fileName = '';
  }

  onClose(): void {
    this.stopPolling();
    // Si la génération est terminée, notifier le dashboard pour refresh des stats
    if (this.view === 'done') {
      this.confirm.emit();
    } else {
      this.close.emit();
    }
  }
}
