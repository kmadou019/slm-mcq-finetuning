/**
 * Models for authentication and user management
 */

export interface User {
  id: string;
  username: string;
  email: string;
  role: 'evaluator' | 'admin';
  created_at: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface MCQAssignment {
  user_id: string;
  mcq_count: number;
  assigned_mcq_ids: string[];
  assigned_at: string;
  status: 'pending' | 'in_progress' | 'completed';
}

export interface MCQSelectionRequest {
  count: number;
  model?: string;
}
