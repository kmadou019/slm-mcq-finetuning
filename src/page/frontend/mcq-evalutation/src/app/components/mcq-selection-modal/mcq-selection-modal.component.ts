import { Component, EventEmitter, Output, signal, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AuthService } from '../../services/auth.service';

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
export class McqSelectionModalComponent {
  private readonly authService = inject(AuthService);

  @Output() close = new EventEmitter<void>();
  @Output() confirm = new EventEmitter<number>();

  // Options prédéfinies
  predefinedOptions = [10, 20, 50, 100];

  // Nombre sélectionné
  selectedCount = signal<number | null>(null);
  customCount = signal<number | null>(null);

  // État
  loading = signal(false);
  error = signal<string | null>(null);

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

    // Validation
    if (!count || count <= 0) {
      this.error.set('Veuillez sélectionner un nombre de questions valide.');
      return;
    }

    if (count > 500) {
      this.error.set('Le nombre maximum de questions est 500.');
      return;
    }

    // Démarrer le chargement
    this.loading.set(true);
    this.error.set(null);

    // Appeler l'API pour assigner les MCQ
    this.authService.assignMCQs(count).subscribe({
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
