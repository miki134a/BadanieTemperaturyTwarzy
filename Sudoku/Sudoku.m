%% Parametry
folder_path = 'S:\ZTDT\Sudoku';
photos_quantity = 3;

% Nie zmieniać poniżej!
vis_path = [folder_path '\Vis'];
termo_path = [folder_path '\Termo'];
calibration_path = [folder_path '\Calibration'];

vis_filename = dir(vis_path);
termo_filename = dir(termo_path);

vis_folder = vis_filename.folder; 
termo_folder =  termo_filename.folder;

vis_img = cell(length(vis_filename)-2,1);
termo_img = cell(length(termo_filename)-2,1);

%% Wczytywanie zdjęć
 for i = 3:1:length(vis_filename)
     vis_img(i-2,1) = {imread([vis_folder '\' vis_filename(i).name])};
     termo_img(i-2,1) = {imread([termo_folder '\' termo_filename(i).name])};
 end

%% Kalibracja temperatury
syms x;
termo = imread([calibration_path '\termo.jpg']);
termo = termo(:,size(termo,2),:);
termo = rgb2gray(termo);
coefficients = polyfit([double(min(termo)), double(max(termo))], [20, 37], 1);
a = coefficients (1);
b = coefficients (2);
f = @(x) a*x+b;
%% Rejestracja obrazów
chess_vis = rgb2gray(imread([calibration_path '\Chessboard_vis.jpg']));
chess_termo = rgb2gray(imread([calibration_path '\Chessboard_termo.jpg']));

[mp,fp] = cpselect(chess_vis,chess_termo,Wait=true);
t = fitgeotform2d(mp,fp,'similarity');
Rfixed = imref2d(size(chess_termo));

for i = 1:length(vis_img)
    vis_img{i} = imwarp(vis_img{i},t,OutputView=Rfixed);
    termo_img{i} = rgb2gray(termo_img{i});
end

%% Punkty charakterystyczne twarzy
load 'DeepPupilNet.mat';

for i = 1:length(vis_img)

cords = eye_localization(vis_img{i},net);

faceDetector = vision.CascadeObjectDetector('Nose'); 
bboxes = step(faceDetector, vis_img{i});
face = vis_img{i}(bboxes(1,2):bboxes(1,2)+bboxes(1,4),bboxes(1,1):bboxes(1,1)+bboxes(1,3));
ftrs = detectMinEigenFeatures(face);
ftrs = selectStrongest(ftrs,2);

y1 = ftrs.Location(2) + bboxes(1,2);
y2 = ftrs.Location(4) + bboxes(1,2);

nose_y = mean([y1 y2]);

eye_x = mean([cords(1) cords(3)]);
eye_y = mean([cords(2) cords(4)]);

eye1_x = cords(1);
eye2_x = cords(3);

forehead_y = eye_y - (nose_y-eye_y);
chin_y = nose_y + (nose_y-eye_y);

points{i} = cornerPoints([eye1_x eye1_x eye_x eye_x eye_x eye_x eye2_x eye2_x;...
                     forehead_y nose_y nose_y eye_y chin_y forehead_y nose_y forehead_y]');

end

%% Akwizycja temperatury w punktach
temperature = zeros(length(points),8);

for i = 1:length(points)
    temp_points = round(points{i}.Location);
    for j = 1:length(temp_points)
        temperature(i,j) = f(double(termo_img{i}(temp_points(j,1),temp_points(j,2))));
    end
end

%% Porównanie
t = 0:1:(photos_quantity-1)

for i = 1:photos_quantity:size(temperature,1)-(photos_quantity-1)
    figure;
    for j = 1:size(temperature,2)
        plot(t,temperature(i:i+photos_quantity-1,j),'-o');
        hold on;
    end
    xticks(t);
end