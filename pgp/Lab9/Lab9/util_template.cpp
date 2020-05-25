#include <GL/glew.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

void checkErrors(std::string desc) 
{
    GLenum e = glGetError();
    if (e != GL_NO_ERROR) 
    {
        fprintf(stderr, "OpenGL error in \"%s\": %s (%d)\n", desc.c_str(), gluErrorString(e), e);
        exit(20);
    }
}

#include <glm/vec3.hpp> // glm::vec3
#include <glm/vec4.hpp> // glm::vec4
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/gtc/matrix_transform.hpp> 
//glm::mat4 camera(float Translate, glm::vec2 const & Rotate);

const unsigned int window_width = 512;
const unsigned int window_height = 512;

GLuint bufferID;
GLuint progHandle;
GLuint genRenderProg();

const int num_of_verticies = 3;

int initBuffer() 
{ //main() {
    glGenBuffers(1, &bufferID);
    glBindBuffer(GL_ARRAY_BUFFER, bufferID);
    static const GLfloat vertex_buffer_data[] = {
    -0.9f, -0.9f, -0.0f, 1.0f, 0.0f, 0.0f,
     0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
     0.9f,  -0.5f, 0.0f, 0.0f, 0.0f, 1.0f,
    };

    glBufferData(GL_ARRAY_BUFFER, 6 * num_of_verticies * sizeof(float),
        vertex_buffer_data, GL_STATIC_DRAW);
    //glBindBuffer( GL_ARRAY_BUFFER, bufferID);

    return 0;
}


void camera() {
    glm::mat4 Projection = glm::perspective(glm::radians(60.0f),
        (float)window_width / (float)window_height, 0.1f, 0.0f);

    // Camera matrix
    glm::mat4 View = glm::lookAt(
        glm::vec3(3, 1, 1), // Camera is at (4,3,3), in World Space
        glm::vec3(0, 0, 0), // and looks at the origin
        glm::vec3(0, 1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
    );

    glm::mat4 Model = glm::mat4(1.0f);
    glm::mat4 mvp = Projection * View * Model;

    GLuint MatrixID = glGetUniformLocation(progHandle, "MVP");
    glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &mvp[0][0]);
}

void display() 
{
    glBindBuffer( GL_ARRAY_BUFFER, bufferID);

    progHandle = genRenderProg();
    glUseProgram(progHandle);
    camera();

    GLuint vertArray;
    glGenVertexArrays(1, &vertArray);
    glBindVertexArray(vertArray);

    GLint posPtr = glGetAttribLocation(progHandle, "pos");
    glVertexAttribPointer(posPtr, 3, GL_FLOAT, GL_FALSE, 24, 0);
    glEnableVertexAttribArray(posPtr);

    GLint colorPtr = glGetAttribLocation(progHandle, "color");
    glVertexAttribPointer(colorPtr, 3, GL_FLOAT, GL_FALSE, 24, (const GLvoid*)12);
    glEnableVertexAttribArray(colorPtr);


    glDrawArrays(GL_POINTS,0, num_of_verticies);
    glDrawArrays(GL_TRIANGLES, 0, num_of_verticies);

    glDisableVertexAttribArray(posPtr);
    glDisableVertexAttribArray(colorPtr);
}

void myCleanup() 
{
    // Cleanup VBO
    glDeleteBuffers(1, &bufferID);
    //glDeleteVertexArrays(1, &VertexArrayID);
    glDeleteProgram(progHandle);
}