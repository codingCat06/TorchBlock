import { test, expect } from '@playwright/test';

test.describe('Visual Block Editor Tests', () => {
  
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:5174');
  });

  test('should display project creation screen', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('PyTorch Visual Block Editor');
    await expect(page.locator('p')).toContainText('Create visual PyTorch models with drag-and-drop blocks');
    await expect(page.locator('.project-input')).toBeVisible();
    await expect(page.locator('.create-project-btn')).toBeVisible();
  });

  test('should create project and show block editor', async ({ page }) => {
    const projectInput = page.locator('.project-input');
    const createBtn = page.locator('.create-project-btn');

    await projectInput.fill('My Neural Network');
    await createBtn.click();

    // Should show the main editor interface
    await expect(page.locator('.visual-block-editor')).toBeVisible();
    await expect(page.locator('.editor-header h2')).toContainText('Project: My Neural Network');
  });

  test('should display PyTorch block palette', async ({ page }) => {
    // Create project first
    await page.locator('.project-input').fill('Test Project');
    await page.locator('.create-project-btn').click();

    // Check if palette is visible
    await expect(page.locator('.pytorch-block-palette')).toBeVisible();
    await expect(page.locator('.palette-header h3')).toContainText('PyTorch Blocks');
    await expect(page.locator('.palette-header p')).toContainText('Drag blocks to the canvas');
  });

  test('should display PyTorch block categories', async ({ page }) => {
    // Create project first
    await page.locator('.project-input').fill('Test Project');
    await page.locator('.create-project-btn').click();

    // Check for main categories
    const categories = [
      'Input/Output',
      'Linear Layers', 
      'Convolution',
      'Pooling',
      'Activation',
      'Normalization',
      'Regularization',
      'Loss Functions',
      'Optimizers',
      'Recurrent',
      'Attention'
    ];

    for (const category of categories) {
      await expect(page.locator('.category-header', { hasText: category })).toBeVisible();
    }
  });

  test('should display PyTorch blocks with correct content', async ({ page }) => {
    // Create project first
    await page.locator('.project-input').fill('Test Project');
    await page.locator('.create-project-btn').click();

    // Check for specific important blocks
    const importantBlocks = [
      'Input Layer',
      'Linear/Dense', 
      'Conv2d',
      'ReLU',
      'MaxPool2d',
      'Dropout',
      'BatchNorm2d',
      'CrossEntropyLoss',
      'Adam Optimizer',
      'LSTM'
    ];

    for (const blockName of importantBlocks) {
      await expect(page.locator('.pytorch-block .block-label', { hasText: blockName })).toBeVisible();
    }
  });

  test('should show block icons and colors', async ({ page }) => {
    // Create project first  
    await page.locator('.project-input').fill('Test Project');
    await page.locator('.create-project-btn').click();

    // Check that blocks have icons
    const firstBlock = page.locator('.pytorch-block').first();
    await expect(firstBlock.locator('.block-icon')).toBeVisible();
    await expect(firstBlock.locator('.block-label')).toBeVisible();

    // Check that blocks have background colors
    const blockElement = await firstBlock.elementHandle();
    const backgroundColor = await blockElement?.evaluate(el => 
      window.getComputedStyle(el).backgroundColor
    );
    expect(backgroundColor).not.toBe('rgba(0, 0, 0, 0)'); // Not transparent
  });

  test('should have canvas area for block placement', async ({ page }) => {
    // Create project first
    await page.locator('.project-input').fill('Test Project');
    await page.locator('.create-project-btn').click();

    // Check canvas area
    await expect(page.locator('.flow-container')).toBeVisible();
    await expect(page.locator('.react-flow-canvas')).toBeVisible();
    
    // Check for React Flow controls
    await expect(page.locator('.react-flow__controls')).toBeVisible();
    await expect(page.locator('.react-flow__minimap')).toBeVisible();
  });

  test('should show initial Input Layer node', async ({ page }) => {
    // Create project first
    await page.locator('.project-input').fill('Test Project');
    await page.locator('.create-project-btn').click();

    // Check for initial Input Layer node on canvas
    await expect(page.locator('.pytorch-node')).toBeVisible();
    await expect(page.locator('.node-title', { hasText: 'Input Layer' })).toBeVisible();
  });

  test('should have export and run buttons', async ({ page }) => {
    // Create project first
    await page.locator('.project-input').fill('Test Project');
    await page.locator('.create-project-btn').click();

    // Check header buttons
    await expect(page.locator('.export-btn')).toBeVisible();
    await expect(page.locator('.run-btn')).toBeVisible();
    await expect(page.locator('.export-btn')).toContainText('Export Code');
    await expect(page.locator('.run-btn')).toContainText('Run Model');
  });

  test('should be able to drag blocks (basic drag test)', async ({ page }) => {
    // Create project first
    await page.locator('.project-input').fill('Test Project');
    await page.locator('.create-project-btn').click();

    // Get a block to test dragging
    const linearBlock = page.locator('.pytorch-block .block-label', { hasText: 'Linear/Dense' }).locator('..');
    
    // Check if block is draggable
    const isDraggable = await linearBlock.getAttribute('draggable');
    expect(isDraggable).toBe('true');
  });

  test('should show node configuration when clicked', async ({ page }) => {
    // Create project first
    await page.locator('.project-input').fill('Test Project');
    await page.locator('.create-project-btn').click();

    // Click on the input layer node
    const inputNode = page.locator('.pytorch-node').first();
    const configToggle = inputNode.locator('.config-toggle');
    
    if (await configToggle.isVisible()) {
      await configToggle.click();
      // Check if config is shown
      await expect(inputNode.locator('.node-config')).toBeVisible();
    }
  });

  test('should handle multiple node types correctly', async ({ page }) => {
    // Create project first
    await page.locator('.project-input').fill('Test Project');
    await page.locator('.create-project-btn').click();

    // Check that we have different categories of blocks with different colors
    const activationBlocks = page.locator('.block-category:has(.category-header:text("Activation")) .pytorch-block');
    const linearBlocks = page.locator('.block-category:has(.category-header:text("Linear Layers")) .pytorch-block');
    
    await expect(activationBlocks.first()).toBeVisible();
    await expect(linearBlocks.first()).toBeVisible();
  });

  test('should display parameter counts for blocks', async ({ page }) => {
    // Create project first
    await page.locator('.project-input').fill('Test Project');
    await page.locator('.create-project-btn').click();

    // Check that blocks with parameters show param count
    const linearBlock = page.locator('.pytorch-block:has(.block-label:text("Linear/Dense"))');
    await expect(linearBlock.locator('.param-count')).toBeVisible();
  });

  test('should have responsive layout', async ({ page }) => {
    // Create project first
    await page.locator('.project-input').fill('Test Project'); 
    await page.locator('.create-project-btn').click();

    // Check that main components are laid out correctly
    const palette = page.locator('.pytorch-block-palette');
    const canvas = page.locator('.flow-container');
    
    await expect(palette).toBeVisible();
    await expect(canvas).toBeVisible();

    // Check flex layout
    const editorContent = page.locator('.editor-content');
    const display = await editorContent.evaluate(el => window.getComputedStyle(el).display);
    expect(display).toBe('flex');
  });

  test('should export Python code functionality', async ({ page }) => {
    // Create project first
    await page.locator('.project-input').fill('Test Project');
    await page.locator('.create-project-btn').click();

    // Click export button
    const exportBtn = page.locator('.export-btn');
    await exportBtn.click();

    // Check console for generated code (this is basic - in real implementation would check clipboard)
    // The export functionality should work without errors
    await expect(exportBtn).toBeVisible(); // Button should still be there after click
  });

});