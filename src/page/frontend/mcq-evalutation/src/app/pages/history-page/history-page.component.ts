import { Component, computed, inject, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
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
  private readonly route = inject(ActivatedRoute);

  loading = signal(true);
  history = signal<any[]>([]);
  error = signal('');
  filter = signal<'all' | 'accepted' | 'rejected'>('all');

  groupedHistory = computed(() => {
    const f = this.filter();
    const entries = this.history().filter(e => {
      if (f === 'accepted') return e.decision === 'ACCEPT';
      if (f === 'rejected') return e.decision === 'REJECT';
      return true;
    });

    const byDay = new Map<string, any[]>();
    for (const entry of entries) {
      const dayKey = entry.validated_at?.slice(0, 10) ?? 'unknown';
      if (!byDay.has(dayKey)) byDay.set(dayKey, []);
      byDay.get(dayKey)!.push(entry);
    }

    return Array.from(byDay.entries())
      .sort(([a], [b]) => b.localeCompare(a))
      .map(([dayKey, dayEntries]) => ({
        label: this.formatDayLabel(dayKey),
        count: dayEntries.length,
        entries: dayEntries,
      }));
  });

  filteredCount = computed(() =>
    this.groupedHistory().reduce((sum, g) => sum + g.count, 0)
  );

  ngOnInit(): void {
    const qp = this.route.snapshot.queryParamMap.get('filter');
    if (qp === 'accepted' || qp === 'rejected') {
      this.filter.set(qp);
    }

    this.mcqService.getValidationHistory(200).subscribe({
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

  setFilter(f: 'all' | 'accepted' | 'rejected'): void {
    this.filter.set(f);
  }

  onReview(entry: any): void {
    this.router.navigate(['/evaluation'], {
      queryParams: { mcq_id: entry.mcq_id, model: entry.model }
    });
  }

  onBack(): void {
    this.router.navigate(['/dashboard']);
  }

  private formatDayLabel(dayKey: string): string {
    if (dayKey === 'unknown') return 'Date inconnue';
    const d = new Date(dayKey + 'T12:00:00');
    return 'Séance du ' + d.toLocaleDateString('fr-FR', {
      weekday: 'long', day: '2-digit', month: 'long', year: 'numeric'
    });
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
