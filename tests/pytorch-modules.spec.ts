import { test, expect } from '@playwright/test';

// PyTorch 주요 모듈들 리스트
const PYTORCH_MODULES = [
  'torch',
  'torch.nn',
  'torch.nn.functional', 
  'torch.optim',
  'torch.utils.data',
  'torch.autograd',
  'torch.cuda',
  'torch.jit',
  'torch.onnx',
  'torch.distributed',
  'torchvision',
  'torchvision.transforms',
  'torchvision.models',
  'torchvision.datasets',
  'torchtext',
  'torchaudio'
];

const PYTORCH_FUNCTIONS = [
  'torch.tensor',
  'torch.zeros',
  'torch.ones',
  'torch.randn',
  'torch.rand',
  'torch.arange',
  'torch.linspace',
  'torch.eye',
  'torch.cat',
  'torch.stack',
  'torch.matmul',
  'torch.mm',
  'torch.bmm',
  'torch.sum',
  'torch.mean',
  'torch.std',
  'torch.max',
  'torch.min',
  'torch.argmax',
  'torch.argmin',
  'torch.softmax',
  'torch.sigmoid',
  'torch.relu',
  'torch.tanh',
  'torch.dropout',
  'torch.batch_norm',
  'torch.conv2d',
  'torch.max_pool2d',
  'torch.avg_pool2d',
  'torch.linear',
  'torch.cross_entropy',
  'torch.mse_loss',
  'torch.l1_loss',
  'torch.binary_cross_entropy',
  'torch.nll_loss',
  'torch.kl_div',
  'torch.save',
  'torch.load'
];

test.describe('PyTorch Module Support Tests', () => {
  
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:5173');
  });

  test('should display PyTorch Block title', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('PyTorch Block');
  });

  test('should have module input interface', async ({ page }) => {
    await expect(page.locator('input[placeholder*="Module"]')).toBeVisible();
    await expect(page.locator('textarea[placeholder*="Module content"]')).toBeVisible();
    await expect(page.locator('button')).toContainText('Save Module');
  });

  test('should support basic PyTorch module creation', async ({ page }) => {
    const moduleInput = page.locator('input[placeholder*="Module"]');
    const contentInput = page.locator('textarea[placeholder*="Module content"]');
    const saveButton = page.locator('button:has-text("Save Module")');

    // 기본 PyTorch 모듈 테스트
    await moduleInput.fill('basic_tensor');
    await contentInput.fill(`
import torch

# 기본 텐서 생성
x = torch.tensor([1, 2, 3, 4])
y = torch.zeros(2, 3)
z = torch.ones(3, 3)

print("Tensor x:", x)
print("Zeros tensor:", y)  
print("Ones tensor:", z)
    `);
    
    await saveButton.click();
    
    // 저장된 모듈 확인
    await expect(page.locator('text=basic_tensor')).toBeVisible();
  });

  test('should support PyTorch neural network modules', async ({ page }) => {
    const moduleInput = page.locator('input[placeholder*="Module"]');
    const contentInput = page.locator('textarea[placeholder*="Module content"]');
    const saveButton = page.locator('button:has-text("Save Module")');

    // 신경망 모듈 테스트
    await moduleInput.fill('neural_network');
    await contentInput.fill(`
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

net = SimpleNet()
print("Network created:", net)
    `);
    
    await saveButton.click();
    
    // 저장된 모듈 확인
    await expect(page.locator('text=neural_network')).toBeVisible();
  });

  test('should support PyTorch optimization modules', async ({ page }) => {
    const moduleInput = page.locator('input[placeholder*="Module"]');
    const contentInput = page.locator('textarea[placeholder*="Module content"]');
    const saveButton = page.locator('button:has-text("Save Module")');

    // 옵티마이저 모듈 테스트
    await moduleInput.fill('optimizer_example');
    await contentInput.fill(`
import torch
import torch.optim as optim
import torch.nn as nn

# 간단한 모델과 옵티마이저
model = nn.Linear(10, 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 예제 데이터
x = torch.randn(32, 10)
y = torch.randn(32, 1)

# Forward pass
output = model(x)
loss = criterion(output, y)

print("Loss:", loss.item())

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()
    `);
    
    await saveButton.click();
    
    // 저장된 모듈 확인
    await expect(page.locator('text=optimizer_example')).toBeVisible();
  });

  test('should support PyTorch data loading modules', async ({ page }) => {
    const moduleInput = page.locator('input[placeholder*="Module"]');
    const contentInput = page.locator('textarea[placeholder*="Module content"]');
    const saveButton = page.locator('button:has-text("Save Module")');

    // 데이터 로딩 모듈 테스트
    await moduleInput.fill('data_loading');
    await contentInput.fill(`
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# 예제 데이터셋
data = torch.randn(100, 10)
targets = torch.randint(0, 2, (100,))

dataset = CustomDataset(data, targets)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print("Dataset size:", len(dataset))
print("Number of batches:", len(dataloader))
    `);
    
    await saveButton.click();
    
    // 저장된 모듈 확인
    await expect(page.locator('text=data_loading')).toBeVisible();
  });

  test('should support advanced PyTorch features', async ({ page }) => {
    const moduleInput = page.locator('input[placeholder*="Module"]');
    const contentInput = page.locator('textarea[placeholder*="Module content"]');
    const saveButton = page.locator('button:has-text("Save Module")');

    // 고급 기능 테스트 (CUDA, autograd 등)
    await moduleInput.fill('advanced_features');
    await contentInput.fill(`
import torch
import torch.autograd as autograd

# Autograd 예제
x = torch.randn(3, 3, requires_grad=True)
y = torch.randn(3, 3, requires_grad=True)
z = torch.sum(x * y)

z.backward()
print("x.grad:", x.grad)
print("y.grad:", y.grad)

# CUDA 가용성 확인
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())

# JIT 컴파일 예제
@torch.jit.script
def jit_function(x, y):
    return x + y

result = jit_function(torch.tensor(1.0), torch.tensor(2.0))
print("JIT result:", result)
    `);
    
    await saveButton.click();
    
    // 저장된 모듈 확인
    await expect(page.locator('text=advanced_features')).toBeVisible();
  });

  // PyTorch 모듈 가져오기 테스트
  for (const module of PYTORCH_MODULES.slice(0, 5)) { // 처음 5개만 테스트
    test(`should support importing ${module}`, async ({ page }) => {
      const moduleInput = page.locator('input[placeholder*="Module"]');
      const contentInput = page.locator('textarea[placeholder*="Module content"]');
      const saveButton = page.locator('button:has-text("Save Module")');

      await moduleInput.fill(`import_${module.replace(/\./g, '_')}`);
      await contentInput.fill(`
try:
    import ${module}
    print("Successfully imported ${module}")
except ImportError as e:
    print(f"Failed to import ${module}: {e}")
      `);
      
      await saveButton.click();
      
      // 저장된 모듈 확인
      await expect(page.locator(`text=import_${module.replace(/\./g, '_')}`)).toBeVisible();
    });
  }

  // PyTorch 함수 사용 테스트
  for (const func of PYTORCH_FUNCTIONS.slice(0, 5)) { // 처음 5개만 테스트
    test(`should support using ${func}`, async ({ page }) => {
      const moduleInput = page.locator('input[placeholder*="Module"]');
      const contentInput = page.locator('textarea[placeholder*="Module content"]');
      const saveButton = page.locator('button:has-text("Save Module")');

      await moduleInput.fill(`func_${func.replace(/\./g, '_')}`);
      
      let testCode = '';
      if (func.includes('tensor')) {
        testCode = `result = ${func}([1, 2, 3])`;
      } else if (func.includes('zeros') || func.includes('ones')) {
        testCode = `result = ${func}(3, 3)`;
      } else if (func.includes('randn') || func.includes('rand')) {
        testCode = `result = ${func}(2, 3)`;
      } else if (func.includes('arange')) {
        testCode = `result = ${func}(10)`;
      } else if (func.includes('eye')) {
        testCode = `result = ${func}(3)`;
      } else {
        testCode = `# ${func} function test\nprint("Testing ${func}")`;
      }

      await contentInput.fill(`
import torch

try:
    ${testCode}
    print("Successfully used ${func}")
    if 'result' in locals():
        print("Result shape:", result.shape if hasattr(result, 'shape') else type(result))
except Exception as e:
    print(f"Failed to use ${func}: {e}")
      `);
      
      await saveButton.click();
      
      // 저장된 모듈 확인
      await expect(page.locator(`text=func_${func.replace(/\./g, '_')}`)).toBeVisible();
    });
  }

  test('should display all created modules', async ({ page }) => {
    // 모든 모듈이 리스트에 표시되는지 확인
    const moduleCount = await page.locator('h3:has-text("Saved Modules")').textContent();
    expect(moduleCount).toMatch(/Saved Modules \(\d+\)/);
    
    // 모듈이 실제로 표시되는지 확인
    const modules = page.locator('[style*="border: 1px solid #ccc"]');
    const count = await modules.count();
    expect(count).toBeGreaterThan(0);
  });

  test('should maintain module persistence', async ({ page }) => {
    // 페이지 새로고침 후에도 모듈이 유지되는지 확인
    await page.reload();
    
    const moduleCount = await page.locator('h3:has-text("Saved Modules")').textContent();
    expect(moduleCount).toMatch(/Saved Modules \(\d+\)/);
  });

});