#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: Matheus Freitas Sant'Ana
Disciplina: Computação Gráfica
Data: 11/02/2023
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far

        # Inicializando pilha com identidade
        GL.stack = [np.array([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])]

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir o desenho dos pontos com a cor emissiva (emissiveColor).

        
        total_points = len(point)
        r = 255*colors['emissiveColor'][0]
        g = 255*colors['emissiveColor'][1]
        b = 255*colors['emissiveColor'][2]

        for i in range(0, total_points, 2):
            if 0 < point[i] < GL.width and 0 < point[i+1] < GL.height:
                gpu.GPU.draw_pixel([int(point[i]), int(point[i+1])], gpu.GPU.RGB8, [r, g, b])

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("Polypoint2D : pontos = {0}".format(point)) # imprime no terminal pontos
        # print("Polypoint2D : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo:
        # pos_x = GL.width//2
        # pos_y = GL.height//2
        # gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 0])  # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)
        
    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).
        total_points = int(len(lineSegments))

        for i in range(0, total_points - 2, 2):
            u1, v1 = int(lineSegments[i]), int(lineSegments[i+1])
            u2, v2 = int(lineSegments[i+2]), int(lineSegments[i+3])

            dx =  abs(u2-u1)
            dy = -abs(v2-v1)

            sx = 1 if u1 < u2 else -1
            sy = 1 if v1 < v2 else -1 

            err = dx + dy
            
            while True:
                GL.polypoint2D([u1, v1], colors) 
                e2 = 2*err
                if e2 >= dy:
                    if u1 == u2:
                        break
                    err += dy
                    u1  += sx
                if e2 <= dx:
                    if v1 == v2:
                        break
                    err += dx
                    v1  += sy

        # print("Polyline2D : lineSegments = {0}".format(lineSegments)) # imprime no terminal
        # print("Polyline2D : colors = {0}".format(colors)) # imprime no terminal as cores
        
        # Exemplo:
        # pos_x = GL.width//2
        # pos_y = GL.height//2
        # gpu.GPU.set_pixel(pos_x, pos_y, 255, 0, 0) # altera um pixel da imagem (u, v, r, g, b)

    @staticmethod
    def L(x, y, x0, y0, x1, y1):
        return (y1-y0)*x - (x1-x0)*y + y0*(x1-x0) - x0*(y1-y0)

    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).
        GL.draw_triangle(vertices, colors, two_dimensional=True)

        # total_vertices  = len(vertices)
        # total_triangles = int(total_vertices/6)
        # triangles = np.array_split(vertices, total_triangles)

        # for i in range(total_triangles):
        #     curr_vertices = triangles[i]

        #     GL.polyline2D([curr_vertices[0], curr_vertices[1], curr_vertices[2], curr_vertices[3]], colors)
        #     GL.polyline2D([curr_vertices[2], curr_vertices[3], curr_vertices[4], curr_vertices[5]], colors)
        #     GL.polyline2D([curr_vertices[4], curr_vertices[5], curr_vertices[0], curr_vertices[1]], colors)

        #     x_min = int(min(curr_vertices[0], curr_vertices[2], curr_vertices[4])*0.85)
        #     x_max = int(max(curr_vertices[0], curr_vertices[2], curr_vertices[4])*1.15)
        #     y_min = int(min(curr_vertices[1], curr_vertices[3], curr_vertices[5])*0.85)
        #     y_max = int(max(curr_vertices[1], curr_vertices[3], curr_vertices[5])*1.15)

        #     for i in range(x_min, x_max, 1):
        #         for j in range(y_min, y_max, 1):
        #             # L(x, y) = (y1 – y0)x – (x1 – x0)y + y0(x1 – x0) – x0(y1 – y0)
        #             if GL.L(i, j, curr_vertices[0], curr_vertices[1], curr_vertices[2], curr_vertices[3]) >= 0 and \
        #                GL.L(i, j, curr_vertices[2], curr_vertices[3], curr_vertices[4], curr_vertices[5]) >= 0 and \
        #                GL.L(i, j, curr_vertices[4], curr_vertices[5], curr_vertices[0], curr_vertices[1]) >= 0:
        #                GL.polypoint2D([i, j], colors)


        # print("TriangleSet2D : vertices = {0}".format(vertices)) # imprime no terminal
        # print("TriangleSet2D : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo:
        # gpu.GPU.draw_pixel([6, 8], gpu.GPU.RGB8, [255, 255, 0])  # altera pixel (u, v, tipo, r, g, b)


    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.
        GL.draw_triangle(point, colors, transparency=True)

        # print("View (print triangleSet): \n", GL.view)
        # print("Model (print triangleSet): \n", GL.model)

        # total_vertices  = len(point)
        # total_triangles = int(total_vertices/9)
        # triangles = np.array_split(point, total_triangles)
        
        # for i in range(total_triangles):
        #     curr_vertices = triangles[i]

        #     # Coordenadas do triângulo
        #     ax, ay, az = curr_vertices[0], curr_vertices[1], curr_vertices[2]
        #     bx, by, bz = curr_vertices[3], curr_vertices[4], curr_vertices[5]
        #     cx, cy, cz = curr_vertices[6], curr_vertices[7], curr_vertices[8]

        #     coordinates = np.array([[ax, bx, cx],
        #                             [ay, by, cy],
        #                             [az, bz, cz],
        #                             [1.0, 1.0, 1.0]])

        #     # Multiplicando por matriz de transform e do viewpoint            
        #     coordinates = np.matmul(GL.model, coordinates)
        #     coordinates = np.matmul(GL.view, coordinates)

        #     # Dividindo os valores pela última linha (não-homogênea)
        #     coordinates /= coordinates[-1]

        #     # Criando lista de pontos
        #     points = []
        #     for i in range(3):
        #         points.append(coordinates[0][i])
        #         points.append(coordinates[1][i])
            
        #     GL.triangleSet2D(points, colors)

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("TriangleSet : pontos = {0}".format(point)) # imprime no terminal pontos
        # print("TriangleSet : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        # gpu.GPU.draw_pixel([10, 10, 10], gpu.GPU.RGB8, [0, 255, 255])  # altera pixel

    @staticmethod
    def rotate_quaternion(axis_rotation, angle):
        # Montando matriz quaternions
        q = np.array([axis_rotation[0]*np.sin(angle/2),
                      axis_rotation[1]*np.sin(angle/2),
                      axis_rotation[2]*np.sin(angle/2),
                      np.cos(angle/2)])
        
        # Normalizando q
        q = q/np.linalg.norm(q)

        i, j, k, r = q
        r11 = 1.0-2.0*(j**2 + k**2)
        r12 = 2.0*(i*j - k*r)
        r13 = 2.0*(i*k + j*r)
        r14 = 0.0
        r21 = 2.0*(i*j + k*r)
        r22 = 1.0 - 2.0*(i**2 + k**2)
        r23 = 2.0*(j*k - i*r)
        r24 = 0.0
        r31 = 2.0*(i*k - j*r)
        r32 = 2.0*(j*k + i*r)
        r33 = 1.0 - 2.0*(i**2 + j**2)
        r34 = 0.0
        r41 = 0.0
        r42 = 0.0
        r43 = 0.0
        r44 = 1.0

        # Matriz de rotação em quatérnions
        R = np.array([[r11, r12, r13, r14],
                      [r21, r22, r23, r24],
                      [r31, r32, r33, r34],
                      [r41, r42, r43, r44]])
        
        return R

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

        # LookAt
        # Matriz Rotação
        R = GL.rotate_quaternion(orientation[:3], orientation[3])
        
        # Matriz translação (não homogênea)
        T_id = np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0],
                         position])
        
        # Fazendo transposição da matriz
        T = T_id.transpose()
        
        # Tornando matriz de translação homogênea
        T = np.append(T,np.array(([[0.0, 0.0, 0.0, 1.0]])), axis=0)
        
        # Matriz lookat
        LookAt = np.linalg.inv(np.matmul(T, R))

        # Definindo variável global
        GL.lookat = LookAt

        # print("LookAt: \n", LookAt)
        
        # Perspectiva
        Fovy = 2.0*np.arctan(np.tan(fieldOfView/2.0)*GL.height/np.sqrt(GL.height**2+GL.width**2))

        top = GL.near*np.tan(Fovy)
        bottom = -top
        right = top * GL.width/GL.height
        left = -right

        p11 = GL.near/right
        p12 = 0.0
        p13 = 0.0
        p14 = 0.0
        p21 = 0.0
        p22 = GL.near/top
        p23 = 0.0
        p24 = 0.0
        p31 = 0.0
        p32 = 0.0
        p33 = -((GL.far+GL.near)/(GL.far-GL.near))
        p34 = -2.0*GL.far*GL.near/(GL.far-GL.near)
        p41 = 0.0
        p42 = 0.0
        p43 = -1.0
        p44 = 0.0
        Perspective = np.array([[p11, p12, p13, p14],
                      [p21, p22, p23, p24],
                      [p31, p32, p33, p34],
                      [p41, p42, p43, p44]])
        
        # Screen
        Screen = np.array([[GL.width/2.0, 0.0, 0.0, GL.width/2.0],
                           [0.0, -GL.height/2.0, 0.0, GL.height/2.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])
        

        # View
        GL.view = np.matmul(Perspective, LookAt)
        GL.view = np.matmul(Screen, GL.view)

        # print("Perspective: \n", Perspective)
        # print("Screen: \n", Screen)
        # print("View: \n", GL.view)
    
        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("Viewpoint : ", end='')
        # print("position = {0} ".format(position), end='')
        # print("orientation = {0} ".format(orientation), end='')
        # print("fieldOfView = {0} ".format(fieldOfView))

    @staticmethod
    def pushmatrix(m):
        """Função para empilhar uma matriz"""
        GL.stack.append(np.matmul(GL.stack[-1], m))

    @staticmethod
    def popmatrix():
        """Função para desempilhar uma matriz"""
        return GL.stack.pop()

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo em alguma estrutura de pilha.
        
        # Matriz translação
        T = np.array([[1.0, 0.0, 0.0, translation[0]],
                      [0.0, 1.0, 0.0, translation[1]],
                      [0.0, 0.0, 1.0, translation[2]],
                      [0.0, 0.0, 0.0, 1.0]])
        
        # Matriz escala
        S = np.array([[scale[0], 0.0, 0.0, 0.0],
                      [0.0, scale[1], 0.0, 0.0],
                      [0.0, 0.0, scale[2], 0.0],
                      [0.0, 0.0, 0.0, 1.0]])

        # Matriz rotação
        R = GL.rotate_quaternion(rotation[:3], rotation[3])
        
        # Realizando transformações
        transform = np.matmul(R, S)
        transform = np.matmul(T, transform)

        # Armazenando na pilha
        GL.pushmatrix(transform)

        # Definindo matriz do model
        GL.model = GL.stack[-1]

        # print("Pilha (in): \n", GL.stack)

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("Transform : ", end='')
        # if translation:
        #     print("translation = {0} ".format(translation), end='') # imprime no terminal
        # if scale:
        #     print("scale = {0} ".format(scale), end='') # imprime no terminal
        # if rotation:
        #     print("rotation = {0} ".format(rotation), end='') # imprime no terminal
        # print("")

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        # Recuperando transformação do modelo
        GL.model = GL.popmatrix()

        # print("Pilha (out): \n", GL.stack)

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("Saindo de Transform")
    
    @staticmethod
    def baricentric(x, y, xa, ya, xb, yb, xc, yc):
        """Função para calcular coordenadas baricêntricas"""

        # alpha = L_BC(x, y)/L_BC(xA, yA)
        alpha = GL.L(x, y, xb, yb, xc, yc)/GL.L(xa, ya, xb, yb, xc, yc)

        # beta = L_CA(x, y)/L_CA(xb, yb)
        beta = GL.L(x, y, xc, yc, xa, ya)/GL.L(xb, yb, xc, yc, xa, ya)

        # gama = L_AB(x, y)/L_AB(xc, yc)
        # gama = L(x, y, xa, ya, xb, yb)/L(xc, yc, xa, ya, xb, yb)

        # gama = 1 - alpha - beta
        gama = 1 - alpha - beta

        # C = alpha*A + beta*B + gama*C
        return alpha, beta, gama
    
    @staticmethod
    def is_inside(i, j, xa, ya, xb, yb, xc, yc):
        """Função para verificar se um ponto está dentro de um triângulo"""

        # Verifica normais para baixo ou para cima (+ ou -)
        return (GL.L(i, j, xa, ya, xb, yb) >= 0 and  GL.L(i, j, xb, yb, xc, yc) >= 0 and \
                GL.L(i, j, xc, yc, xa, ya) >= 0) or (GL.L(i, j, xa, ya, xb, yb) <= 0 and \
                GL.L(i, j, xb, yb, xc, yc) <= 0 and  GL.L(i, j, xc, yc, xa, ya) <= 0)
    
    @staticmethod
    def super_sampling(i, j, xa, ya, xb, yb, xc, yc):
        # 2x2 Supersampling 
        i = int(i*2)
        j = int(j*2)
        _ss = 0

        if GL.is_inside(i, j, int(xa*2), int(ya*2), int(xb*2), int(yb*2), int(xc*2), int(yc*2)):
            _ss += 1

        if GL.is_inside(i+1, j, int(xa*2), int(ya*2), int(xb*2), int(yb*2), int(xc*2), int(yc*2)):
            _ss += 1

        if GL.is_inside(i, j+1, int(xa*2), int(ya*2), int(xb*2), int(yb*2), int(xc*2), int(yc*2)):
            _ss += 1

        if GL.is_inside(i+1, j+1, int(xa*2), int(ya*2), int(xb*2), int(yb*2), int(xc*2), int(yc*2)):
            _ss += 1

        _ss /= 4

        return _ss
        
    @staticmethod
    def draw_pixel_custom_2D(i, j, xa, ya, xb, yb, xc, yc, color, colors, ss=1):
        """Função para desenhar um pixel em 2D"""

        if color is not None: # Se existe definição para interpolar cores
            if 0 < i < GL.width and 0 < j < GL.height: # Se está dentro do FrameBuffer
                # Cálculo das interpolações
                alpha, beta, gama = GL.baricentric(i, j, xa, ya, xb, yb, xc, yc)

                # Previous Color
                prev_color = gpu.GPU.read_pixel([i, j], gpu.GPU.RGB8)*colors['transparency']
                
                # Cor interpolada
                r, g, b = alpha*color[:, 0] + beta*color[:, 1] + gama*color[:, 2]
                r *= (1-colors['transparency'])
                g *= (1-colors['transparency'])
                b *= (1-colors['transparency'])
                new_color = [r, g, b]

                # Combinando as cores
                r, g, b = prev_color + new_color

                gpu.GPU.draw_pixel([i, j], gpu.GPU.RGB8, [r*255, g*255, b*255]) 
        else:
            if 0 < i < GL.width and 0 < j < GL.height: # Se está dentro do FrameBuffer
                    # Previous Color
                    prev_color = gpu.GPU.read_pixel([i, j], gpu.GPU.RGB8)*colors['transparency']
                    
                    # New Color
                    r = colors['emissiveColor'][0]*(1-colors['transparency'])
                    g = colors['emissiveColor'][1]*(1-colors['transparency'])
                    b = colors['emissiveColor'][2]*(1-colors['transparency'])
                    new_color = [r, g, b]

                    # Combinando as cores
                    r, g, b = prev_color + new_color

                    gpu.GPU.draw_pixel([i, j], gpu.GPU.RGB8, [r*255*ss, g*255*ss, b*255*ss]) 

    @staticmethod
    def draw_pixel_custom_3D(i, j, xa, ya, za, xb, yb, zb, xc, yc, zc, color, colors, ss=1):
        """Função para desenhar um pixel em 3D"""

        # Cálculo das interpolações
        alpha, beta, gama = GL.baricentric(i, j, xa, ya, xb, yb, xc, yc)

        # Cálculo do Z interpolado do ponto amostrado 
        Z = 1/(alpha/za + beta/zb + gama/zc)

        if color is not None: # Se existe definição para interpolar cores
            if 0 < i < GL.width and 0 < j < GL.height: # Se está dentro do FrameBuffer
                if(Z < gpu.GPU.read_pixel([i, j], gpu.GPU.DEPTH_COMPONENT32F)): # Se o ponto está mais próximo
                    # Define coordenada Z como ponto mais próximo no Framebuffer de profundidade
                    gpu.GPU.draw_pixel([i, j], gpu.GPU.DEPTH_COMPONENT32F, [Z])

                    # Previous Color
                    prev_color = gpu.GPU.read_pixel([i, j], gpu.GPU.RGB8)*colors['transparency']
                    
                    # Cor interpolada (levando em conta deformação da perspectiva)
                    r, g, b = Z*(alpha*color[:, 0]/za + beta*color[:, 1]/zb + gama*color[:, 2]/zc)
                    r *= (1-colors['transparency'])*255
                    g *= (1-colors['transparency'])*255
                    b *= (1-colors['transparency'])*255

                    # Seta que as cores estejam no intervalo entre 0 e 255
                    r = max(min(r, 255.0), 0.0)
                    g = max(min(g, 255.0), 0.0)
                    b = max(min(b, 255.0), 0.0)

                    new_color = [r, g, b]

                    # Combinando as cores
                    r, g, b = prev_color + new_color

                    gpu.GPU.draw_pixel([i, j], gpu.GPU.RGB8, [r, g, b]) 
        else:
            if 0 < i < GL.width and 0 < j < GL.height: # Se está dentro do FrameBuffer
                if(Z < gpu.GPU.read_pixel([i, j], gpu.GPU.DEPTH_COMPONENT32F)): # Se o ponto está mais próximo
                    # Define coordenada Z como ponto mais próximo no Framebuffer de profundidade
                    gpu.GPU.draw_pixel([i, j], gpu.GPU.DEPTH_COMPONENT32F, [Z])

                    # Previous Color
                    prev_color = gpu.GPU.read_pixel([i, j], gpu.GPU.RGB8)*colors['transparency']

                    # New Color
                    r = colors['emissiveColor'][0]*(1-colors['transparency'])*255.0
                    g = colors['emissiveColor'][1]*(1-colors['transparency'])*255.0
                    b = colors['emissiveColor'][2]*(1-colors['transparency'])*255.0
                    new_color = [r, g, b]

                    # Combinando as cores
                    r, g, b = prev_color + new_color

                    # r, g, b = color_buffer
                    gpu.GPU.draw_pixel([i, j], gpu.GPU.RGB8, [r, g, b]) 

    @staticmethod
    def draw_triangle(points, colors, color=None, two_dimensional=None, transparency=False):
        if two_dimensional is None:
            # Se for 3D, teremos nove valores (3 para cada vértice)
            total_coord = 9
        else:
            # Se for 2D, teremos 6 valores (2 para cada vértice)
            total_coord = 6

        # Pega o total de triângulos e os separa em uma matriz de triângulos
        total_vertices  = len(points)
        total_triangles = int(total_vertices/total_coord)
        total_triangles = total_triangles if total_triangles != 0 else 1
        triangles = np.array_split(points, total_triangles)
        
        for i in range(total_triangles):
            curr_vertices = triangles[i]

            if two_dimensional is None:
                # Coordenadas do triângulo
                ax, ay, az = curr_vertices[0], curr_vertices[1], curr_vertices[2]
                bx, by, bz = curr_vertices[3], curr_vertices[4], curr_vertices[5]
                cx, cy, cz = curr_vertices[6], curr_vertices[7], curr_vertices[8]

                # Montando matriz de coordenadas
                coordinates = np.array([[ax, bx, cx],
                                        [ay, by, cy],
                                        [az, bz, cz],
                                        [1.0, 1.0, 1.0]])

        
                # Multiplicando por matriz de transform            
                coordinates = np.matmul(GL.model, coordinates)

                # Obtendo matriz do Z para a deformação de perspectiva
                temp_Z = np.matmul(GL.lookat, coordinates)

                # Multiplicando por matriz de view
                coordinates = np.matmul(GL.view, coordinates)

                # Dividindo os valores pela última linha (não-homogênea)
                coordinates /= coordinates[-1]
                temp_Z /= temp_Z[-1]

                # Criando lista de pontos 
                points = []

                # Criando lista para guardar os valores Z
                z_coord = []

                # Criando lista para guardar valores Z NDC (transparência)
                z_NDC = []

                for i in range(3):
                    points.append(coordinates[0][i])
                    points.append(coordinates[1][i])
                    z_NDC.append(coordinates[2][i])
                    z_coord.append(temp_Z[2][i])
                        


            else:
                # Criando lista de pontos 
                points = []
                points.append(curr_vertices[0])
                points.append(curr_vertices[1])
                points.append(curr_vertices[2])
                points.append(curr_vertices[3])
                points.append(curr_vertices[4])
                points.append(curr_vertices[5])

            xa, ya = points[0], points[1]
            xb, yb = points[2], points[3]
            xc, yc = points[4], points[5]

            if two_dimensional is None:
                if not transparency:
                    za, zb, zc = z_coord[0], z_coord[1], z_coord[2]
                else:
                    za, zb, zc = z_NDC[0], z_NDC[1], z_NDC[2]
                    
            # Ordem de conexão 
            connectionPoints = [xa, ya, xb, yb, xc, yc, xa, ya]

            # Desenhando as arestas do triângulo (algoritmo de Breschen)
            for i in range(0, 5, 2):
                u1, v1 = round(connectionPoints[i]), round(connectionPoints[i+1])
                u2, v2 = round(connectionPoints[i+2]), round(connectionPoints[i+3])

                dx =  abs(u2-u1)
                dy = -abs(v2-v1)

                sx = 1 if u1 < u2 else -1
                sy = 1 if v1 < v2 else -1 

                err = dx + dy
                
                while True:
                    # Desenha pixel (3D/2D)
                    if two_dimensional is None:
                        GL.draw_pixel_custom_3D(u1, v1, xa, ya, za, xb, yb, zb, xc, yc, zc, color, colors)
                    else:
                        # Anti aliasing (apenas para exemplo 2D)
                        _ss = GL.super_sampling(u1, v1, xa, ya, xb, yb, xc, yc)
                        GL.draw_pixel_custom_2D(u1, v1, xa, ya, xb, yb, xc, yc, color, colors, ss=_ss)                        

                    e2 = 2*err
                    if e2 >= dy:
                        if u1 == u2:
                            break
                        err += dy
                        u1  += sx
                    if e2 <= dx:
                        if v1 == v2:
                            break
                        err += dx
                        v1  += sy

            # Definindo Bounding Box
            x_min = int(min(xa, xb, xc)*0.85)
            x_max = int(max(xa, xb, xc)*1.15)
            y_min = int(min(ya, yb, yc)*0.85)
            y_max = int(max(ya, yb, yc)*1.15)

            for i in range(x_min, x_max, 1):
                for j in range(y_min, y_max, 1):
                    # L(x, y) = (y1 – y0)x – (x1 – x0)y + y0(x1 – x0) – x0(y1 – y0)
                    # Checa normal em ambos os casos: para fora e para dentro (+ ou -)
                    if GL.is_inside(i, j, xa, ya, xb, yb, xc, yc):
                        # Desenha pixel (3D/2D)
                        if two_dimensional is None:
                            GL.draw_pixel_custom_3D(i, j, xa, ya, za, xb, yb, zb, xc, yc, zc, color, colors)
                        else:
                            # Anti aliasing (apenas para exemplo 2D)
                            _ss = GL.super_sampling(i, j, xa, ya, xb, yb, xc, yc)
                            GL.draw_pixel_custom_2D(i, j, xa, ya, xb, yb, xc, yc, color, colors, ss=_ss)



    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado
        clockwise = False

        # Índice para controlar onde está na lista de pontos
        idx_i = 0
        for count in stripCount:
            for v in range(count - 2):
                # Seta offset a partir do índice em que está dos pontos
                i = v*3
                i += idx_i

                # Desenha triângulo
                if not clockwise:
                    currentVerts = point[i:i+9]
                else:
                    currentVerts = point[i:i+3] + point[i+6:i+9] + point[i+3:i+6]
                
                clockwise = not clockwise
                GL.draw_triangle(currentVerts, colors)

            # Soma quando vértices foram usados
            idx_i += count*3
        
        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("TriangleStripSet : pontos = {0} ".format(point), end='')
        # for i, strip in enumerate(stripCount):
        #     print("strip[{0}] = {1} ".format(i, strip), end='')
        # print("")
        # print("TriangleStripSet : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        # gpu   .GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.
        i = 2
        clockwise = False

        while i < len(index):
            # Conecta os pontos até encontrar o -1
            while index[i] != -1:
                # Desenha triângulo
                if not clockwise:
                    points = [point[index[i-2]*3], point[index[i-2]*3+1], point[index[i-2]*3+2],
                            point[index[i-1]*3], point[index[i-1]*3+1], point[index[i-1]*3+2],
                            point[index[i]*3], point[index[i]*3+1], point[index[i]*3+2]] 
                else:
                    points = [point[index[i-2]*3], point[index[i-2]*3+1], point[index[i-2]*3+2],
                            point[index[i]*3], point[index[i]*3+1], point[index[i]*3+2],
                            point[index[i-1]*3], point[index[i-1]*3+1], point[index[i-1]*3+2]] 
                    
                clockwise = not clockwise
                GL.draw_triangle(points, colors)

                # Avança para próximo vértice a ser conectado
                i += 1
            # Chegou a -1, pula 3 pontos e faz o index a partir daí
            i += 3

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("IndexedTriangleStripSet : pontos = {0}, index = {1}".format(point, index))
        # print("IndexedTriangleStripSet : colors = {0}".format(colors)) # imprime as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        # gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel
    
    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.
        i = 2
        clockwise = False

        # print(coord)
        # print(coordIndex)

        while i < len(coordIndex):
            # p0
            idx_i = i-2
            while coordIndex[i] != -1:
                if not clockwise:
                    # Pontos definidos para cada vértice do triângulo
                    points = [coord[coordIndex[idx_i]*3], coord[coordIndex[idx_i]*3+1], coord[coordIndex[idx_i]*3+2],
                              coord[coordIndex[i-1]*3], coord[coordIndex[i-1]*3+1], coord[coordIndex[i-1]*3+2],
                              coord[coordIndex[i]*3], coord[coordIndex[i]*3+1], coord[coordIndex[i]*3+2]]
                    if colorPerVertex and color is not None:
                        # Cores definidos para cada vértice do triângulo
                        colorsVert = np.asarray([[color[colorIndex[idx_i]*3], color[colorIndex[i-1]*3], color[colorIndex[i]*3]],
                                                [color[colorIndex[idx_i]*3+1], color[colorIndex[i-1]*3+1], color[colorIndex[i]*3+1]],
                                                [color[colorIndex[idx_i]*3+2], color[colorIndex[i-1]*3+2], color[colorIndex[i]*3+2]]] )
                else:
                    # Pontos definidos para cada vértice do triângulo
                    points = [coord[coordIndex[idx_i]*3], coord[coordIndex[idx_i]*3+1], coord[coordIndex[idx_i]*3+2],
                              coord[coordIndex[i]*3], coord[coordIndex[i]*3+1], coord[coordIndex[i]*3+2],
                              coord[coordIndex[i-1]*3], coord[coordIndex[i-1]*3+1], coord[coordIndex[i-1]*3+2]] 
                    if colorPerVertex and color is not None:
                        # Cores definidos para cada vértice do triângulo
                        colorsVert = np.asarray([[color[colorIndex[idx_i]*3], color[colorIndex[i]*3], color[colorIndex[i-1]*3]],
                                                [color[colorIndex[idx_i]*3+1], color[colorIndex[i]*3+1], color[colorIndex[i-1]*3+1]],
                                                [color[colorIndex[idx_i]*3+2], color[colorIndex[i]*3+2], color[colorIndex[i-1]*3+2]]])
                        
                # Inverte sentido de conexão
                clockwise = not clockwise

               
                if colorPerVertex and color is not None:
                    # Desenha triângulo com especificação para interpolação
                    GL.draw_triangle(points, colors, color=colorsVert)
                else:
                    # Desenha triângulo sem especificação para interpolação
                    GL.draw_triangle(points, colors)

                # Avança para próximo vértice a ser conectado
                i += 1
            # Chegou a -1, pula 3 pontos e faz o index a partir daí
            i += 3

        # Os prints abaixo são só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("IndexedFaceSet : ")
        # if coord:
        #     print("\tpontos(x, y, z) = {0}, coordIndex = {1}".format(coord, coordIndex))
        # print("colorPerVertex = {0}".format(colorPerVertex))
        # if colorPerVertex and color and colorIndex:
        #     print("\tcores(r, g, b) = {0}, colorIndex = {1}".format(color, colorIndex))
        # if texCoord and texCoordIndex:
        #     print("\tpontos(u, v) = {0}, texCoordIndex = {1}".format(texCoord, texCoordIndex))
        # if current_texture:
        #     image = gpu.GPU.load_texture(current_texture[0])
        #     print("\t Matriz com image = {0}".format(image))
        #     print("\t Dimensões da image = {0}".format(image.shape))
        # print("IndexedFaceSet : colors = {0}".format(colors))  # imprime no terminal as cores

        # # Exemplo de desenho de um pixel branco na coordenada 10, 10
        # gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("NavigationInfo : headlight = {0}".format(headlight)) # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
