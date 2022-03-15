function varargout = voiceRecognitionGui(varargin)
% VOICERECOGNITIONGUI MATLAB code for voiceRecognitionGui.fig
%      VOICERECOGNITIONGUI, by itself, creates a new VOICERECOGNITIONGUI or raises the existing
%      singleton*.
%
%      H = VOICERECOGNITIONGUI returns the handle to a new VOICERECOGNITIONGUI or the handle to
%      the existing singleton*.
%
%      VOICERECOGNITIONGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in VOICERECOGNITIONGUI.M with the given input arguments.
%
%      VOICERECOGNITIONGUI('Property','Value',...) creates a new VOICERECOGNITIONGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before voiceRecognitionGui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to voiceRecognitionGui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES
% Edit the above text to modify the response to help voiceRecognitionGui
% Begin initialization code - DO NOT EDIT

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @voiceRecognitionGui_OpeningFcn, ...
                   'gui_OutputFcn',  @voiceRecognitionGui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT
    
% --- Executes just before voiceRecognitionGui is made visible.
function voiceRecognitionGui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to voiceRecognitionGui (see VARARGIN)

% Choose default command line output for voiceRecognitionGui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes voiceRecognitionGui wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --- Outputs from this function are returned to the command line.
function varargout = voiceRecognitionGui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
axes(handles.axes1)
imshow('C:\Users\BatuhanÇİMEN\Desktop\OruntuTanimaProjesi\OruntuTanimaProjesi\voiceRecognitionProject\mic1.png');
% --- Executes on button press in firstButton.
%Eğitim
function firstButton_Callback(hObject, eventdata, handles)
% hObject    handle to firstButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
run('voiceRecognition.m');

% --- Executes on button press in secondButton.
%Doğruluk Oranı
function secondButton_Callback(hObject, eventdata, handles)
% hObject    handle to secondButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
var = load('voiceRecognition.mat');
set(handles.text3,'String',"Doğruluk Oranı: %"+(100-var.HataOpt));

% --- Executes on button press in thirdButton.
%Tahmin
function thirdButton_Callback(hObject, eventdata, handles)
% hObject    handle to thirdButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
var = load('voiceRecognition.mat');
set(handles.text5,'String',"Ses kime ait: "+var.sinifYY(:,1));
% --- Executes on button press in exitButton.
%Çıkış
function exitButton_Callback(hObject, eventdata, handles)
% hObject    handle to exitButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
delete(handles.figure1)
clear
