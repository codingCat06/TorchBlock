const API_BASE = 'http://localhost:5000/api';

export const api = {
  async getModules() {
    const response = await fetch(`${API_BASE}/modules`);
    return response.json();
  },

  async saveModule(name: string, content: string) {
    const response = await fetch(`${API_BASE}/modules`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ name, content }),
    });
    return response.json();
  },

  async checkHealth() {
    const response = await fetch(`${API_BASE}/health`);
    return response.json();
  },
};