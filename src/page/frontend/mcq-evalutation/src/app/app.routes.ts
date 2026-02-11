import { Routes } from '@angular/router';
import { LoginPageComponent } from './pages/login-page/login-page.component';
import { DashboardPageComponent } from './pages/dashboard-page/dashboard-page.component';
import { EvaluationPageComponent } from './pages/evaluation-page/evaluation-page.component';
import { AdminDashboardPageComponent } from './pages/admin-dashboard-page/admin-dashboard-page.component';
import { authGuard } from './guards/auth.guard';
import { adminGuard } from './guards/admin.guard';

export const routes: Routes = [
  { path: '', redirectTo: '/dashboard', pathMatch: 'full' },
  { path: 'login', component: LoginPageComponent },
  {
    path: 'dashboard',
    component: DashboardPageComponent,
    canActivate: [authGuard]
  },
  {
    path: 'evaluation',
    component: EvaluationPageComponent,
    canActivate: [authGuard]
  },
  {
    path: 'admin',
    component: AdminDashboardPageComponent,
    canActivate: [authGuard, adminGuard]
  },
  { path: '**', redirectTo: '/dashboard' }
];
