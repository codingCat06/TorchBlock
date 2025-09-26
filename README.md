# PyTorch Block - Module Manager

A full-stack React + TypeScript + Node.js application for managing and sharing code modules.

## Project Structure

```
├── client/          # React + TypeScript frontend
├── server/          # Node.js + TypeScript backend
├── shared/          # Shared types and utilities
└── package.json     # Root workspace configuration
```

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Run development servers:
   ```bash
   npm run dev
   ```

   This starts both frontend (http://localhost:5173) and backend (http://localhost:5000) concurrently.

## Features

- ✅ React + TypeScript frontend with Vite
- ✅ Node.js + Express + TypeScript backend
- ✅ Module saving and sharing capabilities
- ✅ RESTful API endpoints
- ✅ CORS enabled for cross-origin requests
- ✅ Development setup with hot reload

## Database Options

Choose one based on your needs:

1. **PostgreSQL** (Recommended for production)
2. **MongoDB** (Good for flexible document storage)
3. **SQLite** (Great for development)
4. **Firebase/Firestore** (Best for real-time collaboration)

## API Endpoints

- `GET /api/health` - Server health check
- `GET /api/modules` - Get all modules
- `POST /api/modules` - Save a new module

## Scripts

- `npm run dev` - Start both frontend and backend in development mode
- `npm run build` - Build both frontend and backend for production
- `npm run start` - Start production server

## Next Steps

1. Choose and integrate a database
2. Add authentication
3. Implement module sharing features
4. Add module versioning
5. Create module export/import functionality# TorchBlock
