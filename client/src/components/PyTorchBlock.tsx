import React from 'react';

interface PyTorchBlockProps {
  id: string;
  label: string;
  color: string;
  icon: string;
  config: any;
}

export function PyTorchBlock({ id, label, color, icon, config }: PyTorchBlockProps) {
  const onDragStart = (event: React.DragEvent, blockData: any) => {
    event.dataTransfer.setData('application/pytorch-block', JSON.stringify({
      moduleType: blockData.id,
      label: blockData.label,
      config: blockData.config
    }));
  };

  return (
    <div
      className="pytorch-block"
      draggable
      onDragStart={(event) => onDragStart(event, { id, label, config })}
      style={{ 
        backgroundColor: color,
        borderColor: color
      }}
    >
      <div className="block-icon">{icon}</div>
      <div className="block-label">{label}</div>
      <div className="block-params">
        {Object.keys(config).length > 0 && (
          <div className="param-count">
            {Object.keys(config).length} params
          </div>
        )}
      </div>
    </div>
  );
}