clear all; 
clc;

% Configuración de directorios
carpeta_enfermos = 'C:\Users\paht4\Desktop\base de datos\modulo\imagenes\testeo_2022\enfermas';
carpeta_sanos = 'C:\Users\paht4\Desktop\base de datos\modulo\imagenes\testeo_2022\sanas';

% Parámetros de procesamiento
tamano_imagen = [150, 200]; % Cambia a la escala deseada

ruta_guardado_mat = 'C:\Users\paht4\Desktop\base de datos\modulo\imagenes\testeo_2022\cargadas\datos_testeo.mat';
ruta_guardado_csv = 'C:\Users\paht4\Desktop\base de datos\modulo\imagenes\testeo_2022\cargadas\etiquetas_testeo.csv';

% Obtener listas de archivos
archivos_enfermos = dir(fullfile(carpeta_enfermos, '*.jpg')); % Cambia la extensión si es diferente
archivos_sanos = dir(fullfile(carpeta_sanos, '*.jpg'));

% Inicializar matrices para almacenar los vectores
num_imagenes = length(archivos_enfermos) + length(archivos_sanos);
vector_dim = prod(tamano_imagen); % Tamaño del vector después de escalar
datos = zeros(num_imagenes, vector_dim); % Matriz para almacenar vectores
etiquetas = zeros(num_imagenes, 1); % 0 para sanos, 1 para enfermos

% Procesar imágenes de árboles enfermos
for i = 1:length(archivos_enfermos)
    i
    archivo = fullfile(carpeta_enfermos, archivos_enfermos(i).name);
    img = imread(archivo);
    img_gray = rgb2gray(imresize(img, tamano_imagen)); % Escalar y pasar a escala de grises
    datos(i, :) = img_gray(:)'; % Vectorizar la imagen
    etiquetas(i) = 1; % Marcar como árbol enfermo
end

% Procesar imágenes de árboles sanos
for i = 1:length(archivos_sanos)
    i
    archivo = fullfile(carpeta_sanos, archivos_sanos(i).name);
    img = imread(archivo);
    img_gray = rgb2gray(imresize(img, tamano_imagen)); % Escalar y pasar a escala de grises
    datos(length(archivos_enfermos) + i, :) = img_gray(:)'; % Vectorizar la imagen
    etiquetas(length(archivos_enfermos) + i) = 0; % Marcar como árbol sano
end

% Mezclar los datos
indices = randperm(num_imagenes);
datos_mezclados = datos(indices, :)'; % Transponer para que las columnas sean las imágenes
etiquetas_mezcladas = etiquetas(indices);

% Guardar los datos mezclados en un archivo .mat
save(ruta_guardado_mat, 'datos_mezclados');

% Guardar las etiquetas mezcladas en un archivo .csv
csvwrite(ruta_guardado_csv, etiquetas_mezcladas);

% Mostrar columnas de árboles enfermos en los datos mezclados
columnas_enfermos = find(etiquetas_mezcladas == 1);

% Mostrar resultados
fprintf('Las columnas que corresponden a árboles enfermos son:\n');
disp(columnas_enfermos);

%%
%arreglar
clear all;
clc;

% Configuración de directorios
carpeta_enfermos1 = 'C:\Users\paht4\Desktop\base de datos\modulo\imagenes\03_11_2021\Ground_RGB_Photos\enfermas';
carpeta_enfermos2 = 'C:\Users\paht4\Desktop\base de datos\modulo\imagenes\08_07_2021\Ground_RGB_Photos\Enfermas';
carpeta_sanos1 = 'C:\Users\paht4\Desktop\base de datos\modulo\imagenes\03_11_2021\Ground_RGB_Photos\Healthy';
carpeta_sanos2 = 'C:\Users\paht4\Desktop\base de datos\modulo\imagenes\8_07_2021\Ground_RGB_Photos\Healthy';

% Parámetros de procesamiento
tamano_imagen = [150, 200]; % Cambia a la escala deseada

ruta_guardado_mat = 'C:\Users\paht4\Desktop\base de datos\modulo\imagenes\datos_mezclados_1.mat';
ruta_guardado_csv = 'C:\Users\paht4\Desktop\base de datos\modulo\imagenes\etiquetas_mezcladas_1.csv';

% Obtener listas de archivos
archivos_enfermos = [dir(fullfile(carpeta_enfermos1, '*.jpg')); dir(fullfile(carpeta_enfermos2, '*.jpg'))];
archivos_sanos = [dir(fullfile(carpeta_sanos1, '*.jpg')); dir(fullfile(carpeta_sanos2, '*.jpg'))];

% Inicializar matrices para almacenar los vectores
num_imagenes = length(archivos_enfermos) + length(archivos_sanos);
vector_dim = prod(tamano_imagen); % Tamaño del vector después de escalar
datos = zeros(num_imagenes, vector_dim); % Matriz para almacenar vectores
etiquetas = zeros(num_imagenes, 1); % 0 para sanos, 1 para enfermos

% Procesar imágenes de árboles enfermos
for i = 1:length(archivos_enfermos)
    archivo = fullfile(archivos_enfermos(i).folder, archivos_enfermos(i).name);
    img = imread(archivo);
    img_gray = rgb2gray(imresize(img, tamano_imagen)); % Escalar y pasar a escala de grises
    datos(i, :) = img_gray(:)'; % Vectorizar la imagen
    etiquetas(i) = 1; % Marcar como árbol enfermo
end

% Procesar imágenes de árboles sanos
for i = 1:length(archivos_sanos)
    archivo = fullfile(archivos_sanos(i).folder, archivos_sanos(i).name);
    img = imread(archivo);
    img_gray = rgb2gray(imresize(img, tamano_imagen)); % Escalar y pasar a escala de grises
    datos(length(archivos_enfermos) + i, :) = img_gray(:)'; % Vectorizar la imagen
    etiquetas(length(archivos_enfermos) + i) = 0; % Marcar como árbol sano
end

% Mezclar los datos
indices = randperm(num_imagenes);
datos_mezclados = datos(indices, :)'; % Transponer para que las columnas sean las imágenes
etiquetas_mezcladas = etiquetas(indices);

% Guardar los datos mezclados en un archivo .mat
save(ruta_guardado_mat, 'datos_mezclados');

% Guardar las etiquetas mezcladas en un archivo .csv
csvwrite(ruta_guardado_csv, etiquetas_mezcladas);

% Mostrar columnas de árboles enfermos en los datos mezclados
columnas_enfermos = find(etiquetas_mezcladas == 1);

% Mostrar resultados
fprintf('Las columnas que corresponden a árboles enfermos son:\n');
disp(columnas_enfermos);

%%
% Directorio donde están las imágenes

% Directorio donde están las imágenes
folder = 'C:\Users\paht4\Desktop\base de datos\modulo\imagenes\testeo_2022\13_07_2022\enfermas';  % Cambia esta ruta a la de tu carpeta de imágenes
outputFolder = 'C:\Users\paht4\Desktop\base de datos\modulo\imagenes\testeo_2022\enfermas';  % Cambia esta ruta según corresponda

% Lista de archivos de imagen en la carpeta
fileList = dir(fullfile(folder, '*.jpg'));  % Cambia la extensión según las imágenes (.jpg, .png, etc.)
totalImages = length(fileList);

if totalImages == 0
    error('No se encontraron imágenes en la carpeta especificada.');
end

% Número inicial para el conteo
startNumber = 1624;  % Cambia este número según el inicio deseado

% Crear el directorio de salida si no existe
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Contador para las imágenes omitidas
skippedImages = 0;

% Renombrar y guardar cada imagen
for i = 1:totalImages
    try
        % Leer el archivo de imagen
        img = imread(fullfile(folder, fileList(i).name));
    catch
        warning('No se pudo cargar la imagen: %s. Se omitirá.', fileList(i).name);
        skippedImages = skippedImages + 1;
        continue;
    end

    % Verificar que la imagen no esté vacía
    if isempty(img)
        warning('La imagen %s está vacía. Se omitirá.', fileList(i).name);
        skippedImages = skippedImages + 1;
        continue;
    end

    % Crear el nuevo nombre usando un número de 5 dígitos
    newName = sprintf('%05d.jpg', startNumber + i - 1);  % Ajusta el número inicial

    % Guardar la imagen con el nuevo nombre en la carpeta de destino
    try
        imwrite(img, fullfile(outputFolder, newName));
    catch
        warning('No se pudo guardar la imagen: %s. Se omitirá.', newName);
        skippedImages = skippedImages + 1;
    end
end

% Mostrar resumen de procesamiento
disp(['Renombrado completado. Total de imágenes omitidas: ', num2str(skippedImages)]);


%%
% Directorio donde están las imágenes
folder = 'C:\Users\paht4\Desktop\base de datos\modulo\imagenes\testeo_2022\26_05_2022\enfermas';  % Cambia esta ruta a la de tu carpeta de imágenes
outputFolder = 'C:\Users\paht4\Desktop\base de datos\modulo\imagenes\testeo_2022\enfermas';  % Cambia esta ruta según corresponda

% Lista de archivos de imagen en la carpeta
fileList = dir(fullfile(folder, '*.jpg'));  % Cambia la extensión según las imágenes (.jpg, .png, etc.)
totalImages = length(fileList);

if totalImages == 0
    error('No se encontraron imágenes en la carpeta especificada.');
end

% Número inicial para el conteo
startNumber = 1687;  % Cambia este número según el inicio deseado

% Crear el directorio de salida si no existe
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Contador para las imágenes omitidas
skippedImages = 0;

% Renombrar, rotar y guardar cada imagen
for i = 1:totalImages
    try
        % Leer el archivo de imagen
        img = imread(fullfile(folder, fileList(i).name));

        % Verificar que la imagen no esté vacía
        if isempty(img)
            warning('La imagen %s está vacía. Se omitirá.', fileList(i).name);
            skippedImages = skippedImages + 1;
            continue;
        end

        % Rotar la imagen 90 grados a la derecha
        rotatedImg = imrotate(img, -90);  % Rotación en sentido horario

        % Crear el nuevo nombre usando un número de 5 dígitos
        newName = sprintf('%05d.jpg', startNumber + i - 1);  % Ajusta el número inicial

        % Generar la ruta completa
        savePath = fullfile(outputFolder, newName);

        % Guardar la imagen rotada
        imwrite(rotatedImg, savePath);

    catch ME
        % Contar la imagen omitida y mostrar el error
        warning('No se pudo procesar la imagen %s. Error: %s', fileList(i).name, ME.message);
        skippedImages = skippedImages + 1;
    end
end

% Mostrar resumen de procesamiento
disp(['Renombrado y rotación completados. Total de imágenes omitidas: ', num2str(skippedImages)]);





