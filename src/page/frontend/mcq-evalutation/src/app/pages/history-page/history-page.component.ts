import { Component, inject, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { McqService } from '../../services/mcq.service';

@Component({
  selector: 'app-history-page',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './history-page.component.html',
  styleUrl: './history-page.component.scss'
})
export class HistoryPageComponent implements OnInit {
  private readonly mcqService = inject(McqService);
  readonly router = inject(Router);

  loading = signal(true);
  history = signal<any[]>([]);
  error = signal('');

  ngOnInit(): void {
    this.mcqService.getValidationHistory(100).subscribe({
      next: (data) => {
        this.history.set(data);
        this.loading.set(false);
      },
      error: () => {
        this.error.set('Impossible de charger l\'historique.');
        this.loading.set(false);
      }
    });
  }

  onReview(entry: any): void {
    this.router.navigate(['/evaluation'], {
      queryParams: { mcq_id: entry.mcq_id, model: entry.model }
    });
  }

  onBack(): void {
    this.router.navigate(['/dashboard']);
  }

  formatDate(dateStr: string): string {
    if (!dateStr) return '-';
    const d = new Date(dateStr);
    return d.toLocaleDateString('fr-FR', {
      day: '2-digit', month: '2-digit', year: 'numeric',
      hour: '2-digit', minute: '2-digit'
    });
  }

  truncate(text: string, max: number = 100): string {
    if (!text) return '-';
    return text.length > max ? text.slice(0, max) + '…' : text;
  }
}
