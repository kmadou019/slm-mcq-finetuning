import { Component, EventEmitter, Output, signal, inject, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { McqService } from '../../services/mcq.service';

type ModalView = 'form' | 'generating' | 'done' | 'error';

export const DEFAULT_PROMPT = `À partir du contenu éducatif suivant, générez exactement deux questions à choix multiple avec quatre options de réponse chacune (a, b, c, d), dont une seule est correcte.

OBJECTIFS :
- Les questions doivent évaluer la compréhension des idées principales.
- Les distracteurs doivent être plausibles mais incorrects.
- Les options doivent être courtes.
- Fournir une justification pédagogique pour chaque option.
- Fournir un commentaire global pour chaque question.

CONTRAINTES STRICTES DE SORTIE :
1. La sortie doit être STRICTEMENT un unique objet JSON valide.
2. Interdiction ABSOLUE d'ajouter :
   - des blocs \`\`\`json
   - plusieurs objets JSON
   - du texte avant ou après le JSON
   - des explications hors champs JSON
3. Les champs "correct_option1" et "correct_option2" doivent contenir EXACTEMENT une lettre minuscule parmi : "a", "b", "c", "d".
4. Utiliser uniquement des doubles quotes : "..."
5. Le JSON doit contenir EXACTEMENT les 22 champs suivants :

{
  "question1": "...",
  "question1_comment": "...",
  "option_a1": "...",
  "option_a1_comment": "...",
  "option_b1": "...",
  "option_b1_comment": "...",
  "option_c1": "...",
  "option_c1_comment": "...",
  "option_d1": "...",
  "option_d1_comment": "...",
  "correct_option1": "a",
  "question2": "...",
  "question2_comment": "...",
  "option_a2": "...",
  "option_a2_comment": "...",
  "option_b2": "...",
  "option_b2_comment": "...",
  "option_c2": "...",
  "option_c2_comment": "...",
  "option_d2": "...",
  "option_d2_comment": "...",
  "correct_option2": "c"
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

  @Output() close = new EventEmitter<void>();
  @Output() confirm = new EventEmitter<void>();

  // Modèles disponibles
  availableModels = signal<{ model: string, count: number, available: number }[]>([]);
  loadingModels = signal(true);

  // Champs du formulaire (propriétés simples pour ngModel)
  content = '';
  selectedModel = 'llama3_1_8b';
  promptTemplate = DEFAULT_PROMPT;

  // Prompt editor
  showPromptEditor = false;

  // État de la vue
  view: ModalView = 'form';
  errorMessage = '';
  mcqCount = 0;

  private jobId: string | null = null;
  private pollingIntervalId: any = null;

  get charCount(): number {
    return this.content.length;
  }

  get isContentValid(): boolean {
    return this.content.trim().length >= 100;
  }

  get isPromptModified(): boolean {
    return this.promptTemplate !== DEFAULT_PROMPT;
  }

  ngOnInit(): void {
    this.mcqService.getAvailableModels().subscribe({
      next: (models) => {
        this.availableModels.set(models);
        if (models.length > 0) {
          this.selectedModel = models[0].model;
        }
        this.loadingModels.set(false);
      },
      error: () => {
        const fallback = [
          'llama3_1_8b', 'gemma2_9b', 'medGemma_4b', 'medGemma_27b',
          'qwen3_8b', 'mistral_7b', 'eurollm_9b', 'qwen3_4b_pdapt_slerp'
        ].map(model => ({ model, count: 0, available: 0 }));
        this.availableModels.set(fallback);
        this.loadingModels.set(false);
      }
    });
  }

  ngOnDestroy(): void {
    this.stopPolling();
  }

  togglePromptEditor(): void {
    this.showPromptEditor = !this.showPromptEditor;
  }

  resetPrompt(): void {
    this.promptTemplate = DEFAULT_PROMPT;
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
          } else if (status.status === 'error') {
            this.stopPolling();
            this.view = 'error';
            this.errorMessage = status.error || 'La génération a échoué.';
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

  onRetry(): void {
    this.stopPolling();
    this.view = 'form';
    this.errorMessage = '';
    this.jobId = null;
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
