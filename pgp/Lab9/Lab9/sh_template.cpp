#include <GL/glew.h>
#include <stdio.h>
#include <string>
#include <stdlib.h>

void checkErrors(std::string desc);

GLuint genRenderProg() 
{
    GLuint progHandle = glCreateProgram();
    GLuint vp = glCreateShader(GL_VERTEX_SHADER);
    GLuint fp = glCreateShader(GL_FRAGMENT_SHADER);

    const char* vpSrc[] = {
        "#version 430\n",
        "layout(location = 0) in vec3 pos;\
		 layout(location = 1) in vec3 color;\
		 out vec4 vs_color;\
		 uniform mat4 MVP;\
		 void main() {\
		    gl_Position = MVP*vec4(pos,1);\
		    vs_color=vec4(color,1.0);\
		 }"
    };

    const char* fpSrc[] = {
        "#version 430\n",
         "in vec4 vs_color;\
		  out vec4 fcolor;\
		  void main() {\
			 fcolor = vs_color;\
		 }"
    };

    glShaderSource(vp, 2, vpSrc, NULL);
    glShaderSource(fp, 2, fpSrc, NULL);

    glCompileShader(vp);
    int rvalue;
    glGetShaderiv(vp, GL_COMPILE_STATUS, &rvalue);
    if (!rvalue) 
    {
        fprintf(stderr, "Error in compiling vp\n");
        exit(30);
    }
    glAttachShader(progHandle, vp);


    glCompileShader(fp);
    glGetShaderiv(fp, GL_COMPILE_STATUS, &rvalue);
    if (!rvalue) 
    {
        fprintf(stderr, "Error in compiling fp\n");
        exit(31);
    }
    glAttachShader(progHandle, fp);

    //glBindFragDataLocation(progHandle, 0, "color");
    glLinkProgram(progHandle);

    glGetProgramiv(progHandle, GL_LINK_STATUS, &rvalue);
    if (!rvalue) 
    {
        fprintf(stderr, "Error in linking sp\n");
        exit(32);
    }

    checkErrors("Render shaders");

    return progHandle;
}
