import { useState, useEffect } from 'react';
import { api } from '../services/api';

export function ModuleManager() {
  const [modules, setModules] = useState([]);
  const [newModuleName, setNewModuleName] = useState('');
  const [newModuleContent, setNewModuleContent] = useState('');

  useEffect(() => {
    loadModules();
  }, []);

  const loadModules = async () => {
    try {
      const data = await api.getModules();
      setModules(data.modules);
    } catch (error) {
      console.error('Failed to load modules:', error);
    }
  };

  const saveModule = async () => {
    if (!newModuleName || !newModuleContent) return;
    
    try {
      await api.saveModule(newModuleName, newModuleContent);
      setNewModuleName('');
      setNewModuleContent('');
      loadModules();
    } catch (error) {
      console.error('Failed to save module:', error);
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h2>Module Manager</h2>
      
      <div style={{ marginBottom: '20px' }}>
        <h3>Create New Module</h3>
        <input
          type="text"
          placeholder="Module name"
          value={newModuleName}
          onChange={(e) => setNewModuleName(e.target.value)}
          style={{ marginRight: '10px', padding: '8px' }}
        />
        <br /><br />
        <textarea
          placeholder="Module content"
          value={newModuleContent}
          onChange={(e) => setNewModuleContent(e.target.value)}
          rows={10}
          cols={50}
          style={{ display: 'block', marginBottom: '10px' }}
        />
        <button onClick={saveModule} style={{ padding: '8px 16px' }}>
          Save Module
        </button>
      </div>

      <div>
        <h3>Saved Modules ({modules.length})</h3>
        {modules.length === 0 ? (
          <p>No modules saved yet.</p>
        ) : (
          modules.map((module: any, index) => (
            <div key={index} style={{ border: '1px solid #ccc', padding: '10px', margin: '10px 0' }}>
              <strong>{module.name}</strong>
              <pre style={{ background: '#f5f5f5', padding: '10px', marginTop: '5px' }}>
                {module.content}
              </pre>
            </div>
          ))
        )}
      </div>
    </div>
  );
}