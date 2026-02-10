import { Component, inject, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, ReactiveFormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService } from '../../services/auth.service';
import { MCQCard, SectionCheck } from '../../models';

/**
 * EvaluationPage Component - Page d'évaluation des MCQ
 */
@Component({
  selector: 'app-evaluation-page',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './evaluation-page.component.html',
  styleUrl: './evaluation-page.component.scss'
})
export class EvaluationPageComponent implements OnInit {
  private readonly fb = inject(FormBuilder);
  private readonly authService = inject(AuthService);
  private readonly router = inject(Router);

  // État du composant
  loading = signal(true);
  currentIndex = signal(0);
  currentMcq = signal<MCQCard | null>(null);
  mcqList = signal<string[]>([]); // Liste des IDs de MCQ assignés

  // Formulaire de validation
  validationForm: FormGroup;

  constructor() {
    // Initialiser le formulaire de validation
    this.validationForm = this.fb.group({
      feedback: ['']
    });
  }

  ngOnInit(): void {
    // TODO: Charger les MCQ assignés depuis l'API
    // Pour l'instant, utiliser des données mock
    this.loadAssignedMCQs();
  }

  /**
   * Charger les MCQ assignés (mock pour l'instant)
   */
  private loadAssignedMCQs(): void {
    // TODO: Remplacer par un vrai appel API
    // this.mcqService.getAssignedMCQs().subscribe(...)

    // Mock data - simuler un délai de chargement
    setTimeout(() => {
      this.mcqList.set(['MCQ-000001', 'MCQ-000002', 'MCQ-000003']);
      this.loadMCQ(0);
    }, 500);
  }

  /**
   * Charger un MCQ spécifique
   */
  private loadMCQ(index: number): void {
    this.loading.set(true);
    const mcqId = this.mcqList()[index];

    // TODO: Remplacer par un vrai appel API
    // this.mcqService.getMCQ(mcqId).subscribe(...)

    // Mock data pour démonstration
    setTimeout(() => {
      const mockMCQ: MCQCard = {
        item_id: mcqId,
        source_material: 'Generated from LISA text',
        generator_info: 'GPT-4',
        output_format: 'JSON',
        mcq_question: 'Quelle est la capitale de la France ?',
        options: {
          A: 'Londres',
          B: 'Paris',
          C: 'Berlin',
          D: 'Madrid'
        },
        correct_option: 'B',
        section_a_checks: [
          {
            check_id: 'A1',
            description: 'La question est claire et sans ambiguïté',
            status: 'not_checked',
            confidence: null
          },
          {
            check_id: 'A2',
            description: 'Les options sont distinctes et plausibles',
            status: 'not_checked',
            confidence: null
          },
          {
            check_id: 'A3',
            description: 'La réponse correcte est sans équivoque',
            status: 'not_checked',
            confidence: null
          }
        ],
        section_b_checks: [
          {
            check_id: 'B1',
            description: 'La question est basée sur le texte LISA',
            status: 'not_checked',
            confidence: null
          },
          {
            check_id: 'B2',
            description: 'Le niveau de difficulté est approprié',
            status: 'not_checked',
            confidence: null
          }
        ],
        decision_policy: 'strict',
        final_decision: 'REVISE',
        audit_trail: '',
        lisa_texte_brut: 'Paris est la capitale et la plus grande ville de la France. Elle se situe au centre du Bassin parisien, sur une boucle de la Seine.'
      };

      this.currentMcq.set(mockMCQ);
      this.currentIndex.set(index);
      this.loading.set(false);
    }, 300);
  }

  /**
   * Toggle le statut d'un check (Section A ou B)
   */
  toggleCheck(section: 'A' | 'B', checkIndex: number): void {
    const mcq = this.currentMcq();
    if (!mcq) return;

    const checks = section === 'A' ? mcq.section_a_checks : mcq.section_b_checks;
    const check = checks[checkIndex];

    // Cycle through: not_checked → pass → fail → not_checked
    if (check.status === 'not_checked') {
      check.status = 'pass';
      check.confidence = 'high';
    } else if (check.status === 'pass') {
      check.status = 'fail';
      check.confidence = 'high';
    } else {
      check.status = 'not_checked';
      check.confidence = null;
    }

    // Forcer la mise à jour du signal
    this.currentMcq.set({ ...mcq });
  }

  /**
   * Accepter le MCQ
   */
  onAccept(): void {
    if (confirm('Accepter ce MCQ ?')) {
      this.submitValidation('ACCEPT');
    }
  }

  /**
   * Rejeter le MCQ
   */
  onReject(): void {
    if (confirm('Rejeter ce MCQ ?')) {
      this.submitValidation('REVISE');
    }
  }

  /**
   * Soumettre la validation
   */
  private submitValidation(decision: 'ACCEPT' | 'REVISE'): void {
    const mcq = this.currentMcq();
    if (!mcq) return;

    const validation = {
      mcq_id: mcq.item_id,
      decision: decision,
      section_a_checks: mcq.section_a_checks,
      section_b_checks: mcq.section_b_checks,
      feedback: this.validationForm.get('feedback')?.value || ''
    };

    console.log('Validation soumise:', validation);

    // TODO: Envoyer au backend
    // this.mcqService.submitValidation(validation).subscribe(...)

    // Passer au MCQ suivant ou retourner au dashboard
    if (this.hasNext()) {
      this.onNext();
    } else {
      alert('Toutes les évaluations sont terminées !');
      this.router.navigate(['/dashboard']);
    }
  }

  /**
   * MCQ précédent
   */
  onPrevious(): void {
    if (this.hasPrevious()) {
      this.loadMCQ(this.currentIndex() - 1);
      this.validationForm.reset();
    }
  }

  /**
   * MCQ suivant
   */
  onNext(): void {
    if (this.hasNext()) {
      this.loadMCQ(this.currentIndex() + 1);
      this.validationForm.reset();
    }
  }

  /**
   * Retour au dashboard
   */
  onBackToDashboard(): void {
    if (confirm('Quitter l\'évaluation et retourner au tableau de bord ?')) {
      this.router.navigate(['/dashboard']);
    }
  }

  /**
   * Vérifier si on peut aller au précédent
   */
  hasPrevious(): boolean {
    return this.currentIndex() > 0;
  }

  /**
   * Vérifier si on peut aller au suivant
   */
  hasNext(): boolean {
    return this.currentIndex() < this.mcqList().length - 1;
  }

  /**
   * Obtenir le statut du check (icône)
   */
  getCheckIcon(status: string): string {
    if (status === 'pass') return '✓';
    if (status === 'fail') return '✗';
    return '○';
  }

  /**
   * Obtenir la classe CSS du check
   */
  getCheckClass(status: string): string {
    if (status === 'pass') return 'pass';
    if (status === 'fail') return 'fail';
    return 'not-checked';
  }
}
