import { Component, EventEmitter, Output, signal, inject, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AuthService } from '../../services/auth.service';
import { McqService } from '../../services/mcq.service';

/**
 * MCQSelectionModal Component - Modal pour sélectionner le nombre de MCQ
 */
@Component({
  selector: 'app-mcq-selection-modal',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './mcq-selection-modal.component.html',
  styleUrl: './mcq-selection-modal.component.scss'
})
export class McqSelectionModalComponent implements OnInit {
  private readonly authService = inject(AuthService);
  private readonly mcqService = inject(McqService);

  @Output() close = new EventEmitter<void>();
  @Output() confirm = new EventEmitter<number>();

  // Options prédéfinies
  predefinedOptions = [10, 20, 50, 100];

  // Nombre sélectionné
  selectedCount = signal<number | null>(null);
  customCount = signal<number | null>(null);

  // Modèles disponibles
  availableModels = signal<{ model: string, count: number, available: number }[]>([]);
  selectedModel = signal<string>('qwen3_4b_pdapt_slerp');

  // État
  loading = signal(false);
  loadingModels = signal(true);
  error = signal<string | null>(null);

  ngOnInit(): void {
    // Charger les modèles disponibles depuis le backend
    this.loadingModels.set(true);
    this.mcqService.getAvailableModels().subscribe({
      next: (models) => {
        this.availableModels.set(models);
        this.loadingModels.set(false);
      },
      error: (error) => {
        console.error('Erreur lors du chargement des modèles:', error);
        // Fallback: utiliser les modèles prédéfinis si le backend n'est pas disponible
        const defaultModels = [
          'llama3_1_8b', 'openbiollm_8b', 'gemma2_9b',
          'medGemma_4b', 'medGemma_27b', 'qwen3_8b',
          'mistral_7b', 'eurollm_9b', 'apertus_8B',
          'qwen3_0.6b', 'qwen3_1_7b', 'qwen3_4b',
          'qwen3_8b_pdapt_slerp', 'qwen3_4b_pdapt_slerp',
          'qwen3_1_7b_pdapt_slerp', 'qwen3_0.6b_pdapt_slerp'
        ].map(model => ({ model, count: 0, available: 0 }));
        this.availableModels.set(defaultModels);
        this.loadingModels.set(false);
      }
    });
  }

  /**
   * Sélectionner une option prédéfinie
   */
  selectOption(count: number): void {
    this.selectedCount.set(count);
    this.customCount.set(null);
    this.error.set(null);
  }

  /**
   * Utiliser un nombre personnalisé
   */
  useCustomCount(): void {
    const custom = this.customCount();
    if (custom && custom > 0) {
      this.selectedCount.set(custom);
      this.error.set(null);
    }
  }

  /**
   * Valider la sélection
   */
  onConfirm(): void {
    const count = this.selectedCount();
    const model = this.selectedModel();

    // Validation
    if (!count || count <= 0) {
      this.error.set('Veuillez sélectionner un nombre de questions valide.');
      return;
    }

    if (count > 500) {
      this.error.set('Le nombre maximum de questions est 500.');
      return;
    }

    if (!model) {
      this.error.set('Veuillez sélectionner un modèle.');
      return;
    }

    // Démarrer le chargement
    this.loading.set(true);
    this.error.set(null);

    // Appeler l'API pour assigner les MCQ avec le modèle sélectionné
    this.authService.assignMCQs(count, model).subscribe({
      next: (assignment) => {
        console.log('MCQ assigned:', assignment);
        this.loading.set(false);
        this.confirm.emit(count);
      },
      error: (error) => {
        console.error('Assignment error:', error);
        this.loading.set(false);
        this.error.set('Une erreur est survenue. Veuillez réessayer.');
      }
    });
  }

  /**
   * Fermer le modal
   */
  onClose(): void {
    this.close.emit();
  }
}
