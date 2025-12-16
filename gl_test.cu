// nvcc gl_test.cu -o gl_test -lGL -lGLU -lglut -lGLEW -lcudart
//
// echo $DISPLAY
// export DISPLAY=:0
//
// ./gl_test

// 1. GLEW MUST be first among GL headers to load function pointers correctly.
#include <GL/glew.h>

// 2. CUDA Interop (This header needs GL definitions, so it comes after GLEW).
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

// 3. GLUT (This header relies on basic GL definitions, so it comes after GLEW).
#include <GL/freeglut.h>

#include <iostream>

// --- Configuration ---
const int WIDTH = 7680;
const int HEIGHT = 4320;
const int PIXEL_COUNT = WIDTH * HEIGHT;

// --- Global Variables ---
GLuint pbo;       // Pixel Buffer Object ID (for efficient transfers)
GLuint textureID; // OpenGL Texture ID
cudaGraphicsResource_t pbo_resource; // CUDA-GL resource handle

// --- CUDA Kernel ---

// This kernel generates a simple dynamic visual (a wave pattern)
__global__ void render_kernel(unsigned char *ptr, int width, int height,
                              float time) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    // Calculate the index for the R, G, B, A components
    int index = (y * width + x) * 4;

    // --- Simple Wave Pattern ---
    // Use sine functions and the time variable to create movement
    float r_val = sinf(0.01f * x + time) * 0.5f + 0.5f;
    float g_val = sinf(0.01f * y - time) * 0.5f + 0.5f;
    float b_val = sinf(0.005f * (x + y) + time * 0.5f) * 0.5f + 0.5f;

    // Convert [0, 1] float to [0, 255] byte
    ptr[index + 0] = (unsigned char)(r_val * 255.0f); // R
    ptr[index + 1] = (unsigned char)(g_val * 255.0f); // G
    ptr[index + 2] = (unsigned char)(b_val * 255.0f); // B
    ptr[index + 3] = 255;                             // A (Opaque)
  }
}

// --- OpenGL/CUDA Interop Functions ---

void initGL() {
  // --- ADD THIS LINE ---
  GLenum err = glewInit();
  if (GLEW_OK != err) {
    std::cerr << "GLEW Error: " << glewGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
  // --- END ADDITION ---

  // Enable 2D Textures
  glEnable(GL_TEXTURE_2D);

  // 1. Create a Texture Object
  glGenTextures(1, &textureID);
  glBindTexture(GL_TEXTURE_2D, textureID);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  // Allocate texture memory on the GPU (must be done before creating the PBO)
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, NULL);

  // 2. Create a Pixel Buffer Object (PBO)
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, PIXEL_COUNT * 4 * sizeof(unsigned char),
               NULL, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); // Unbind
}

void initCUDA() {
  // 3. Register the PBO with CUDA for interop
  cudaError_t err = cudaGraphicsGLRegisterBuffer(
      &pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
  if (err != cudaSuccess) {
    std::cerr << "CUDA-GL Registration failed: " << cudaGetErrorString(err)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

// --- Display and Rendering ---

void display() {
  static float frame_time = 0.0f;
  frame_time += 0.01f; // Increment time for animation

  // 1. Map OpenGL PBO to a CUDA device pointer
  unsigned char *d_ptr;
  size_t size;

  // Lock the resource for CUDA access
  cudaGraphicsMapResources(1, &pbo_resource, 0);

  // Get the device pointer and size
  cudaGraphicsResourceGetMappedPointer((void **)&d_ptr, &size, pbo_resource);

  // 2. Launch the CUDA Kernel
  dim3 block(16, 16);
  dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

  render_kernel<<<grid, block>>>(d_ptr, WIDTH, HEIGHT, frame_time);

  // Synchronize to ensure the kernel finishes writing before GL reads
  cudaDeviceSynchronize();

  // 3. Unmap the resource
  cudaGraphicsUnmapResources(1, &pbo_resource, 0);

  // 4. OpenGL Drawing
  glClear(GL_COLOR_BUFFER_BIT);

  // Bind the PBO and specify the data source for the texture
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

  // Copy PBO data into the bound texture
  glBindTexture(GL_TEXTURE_2D, textureID);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA,
                  GL_UNSIGNED_BYTE, 0);

  // Draw a fullscreen quad (2D rectangle) with the texture
  glBegin(GL_QUADS);
  glTexCoord2f(0.0f, 0.0f);
  glVertex2f(-1.0f, -1.0f); // Bottom Left
  glTexCoord2f(1.0f, 0.0f);
  glVertex2f(1.0f, -1.0f); // Bottom Right
  glTexCoord2f(1.0f, 1.0f);
  glVertex2f(1.0f, 1.0f); // Top Right
  glTexCoord2f(0.0f, 1.0f);
  glVertex2f(-1.0f, 1.0f); // Top Left
  glEnd();

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); // Unbind PBO
  glutSwapBuffers();                       // Swap the front and back buffers
}

void cleanup() {
  // Unregister the CUDA resource
  cudaGraphicsUnregisterResource(pbo_resource);

  // Delete GL resources
  glDeleteBuffers(1, &pbo);
  glDeleteTextures(1, &textureID);

  std::cout << "Cleanup complete. Exiting." << std::endl;
}
#include <GL/glew.h> // <-- ADD THIS LINE
void idle() {
  // Request redisplay to trigger the next frame
  glutPostRedisplay();
}

int main(int argc, char **argv) {
  // 1. Initialize GLUT and create window
  glutInit(&argc, argv);

  // --- ADD THESE LINES ---
  // Request a 3.3 core profile context
  glutInitContextVersion(3, 3);
  glutInitContextProfile(GLUT_CORE_PROFILE);
  // --- END ADDITION ---

  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowSize(WIDTH, HEIGHT);
  glutCreateWindow("CUDA-GL Rendering Demo");

  // 2. Setup GL and CUDA
  initGL();
  initCUDA();

  // 3. Register callback functions
  glutDisplayFunc(display);
  glutIdleFunc(idle);
  glutCloseFunc(cleanup); // Ensure cleanup runs on window close

  std::cout << "Starting CUDA-GL Demo. Close window to exit." << std::endl;

  // 4. Start the main loop
  glutMainLoop();

  return 0;
}