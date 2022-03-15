clc;
clear;
%%
DataDizinEgitim='DataEgitim';
DataDizinTest='DataTest';
YSATekrarSayisi=3;
GizliKatmanModSayisi=10;

%% VERİ TABANI OLUŞTURULMASI
% Eğitim Veri tabanı
dbEgitim = datastore(fullfile(pwd,DataDizinEgitim),...
                      'IncludeSubfolders',true,...
                      'FileExtensions','.wav',...
                      'LabelSource','foldernames',...
                      'Type','image');
% Test Veri tabanı
dbTest = datastore(fullfile(pwd,DataDizinTest),...
                      'IncludeSubfolders',true,...
                      'FileExtensions','.wav',...
                      'LabelSource','foldernames',...
                      'Type','image');
% Eğitim ve Test sınıf etiketlerinin tayini
[Siniflar,SinifEgitim,SinifTest]=siniflarinAtanmasi(dbEgitim,dbTest);
clear DataDizin* 

%% YSA EĞİTİMİ
fprintf('\n');
% Eğitim için YSA Giriş-Çıkış verilerinin atanması
xEgitim = Dosya2YSAGirisData(dbEgitim);
yEgitim = Sinif2YSACikisData(Siniflar,SinifEgitim);
% Başlangıç için uydurma değer
HataOpt=1e10;  
for k=1:YSATekrarSayisi
    net = patternnet(GizliKatmanModSayisi);
    net.trainParam.showWindow=false;
    % YSA Eğitilmesi
    net = train(net,xEgitim,yEgitim); 
    % YSA Model çıktısı
    yyEgitim = net(xEgitim);            
    % YSA model çıktısının sınıf değerlerinin çağrılması
    [~,sinifYY] = ysaCikis2Sinif(yyEgitim,Siniflar); 
    % YSA modelinin hata istatistiğinin çağrılması
    [~,HataYuzde] = modelDegerlendirme(SinifEgitim,sinifYY);
    fprintf(1,'Eğitim için YSA Modelleme Tekrar: %02d  > Hata(%%): %.2f\n',k,HataYuzde);
    % Optimum değerlerin kontrolü ve atanması
    if HataYuzde<HataOpt
        netOpt=net;
        HataOpt=HataYuzde;        
    end
    clear net yyEğitim YY sinifYY KiyasSonuc HataYuzde
end
fprintf(1,'\n');
fprintf(1,'Eğitim için elde edilen min hata(%%): %.4f \n\n',HataOpt);
clear k

%% YSA TEST
% Test için YSA Giriş-Çıkış verilerinin atanması
xTest = Dosya2YSAGirisData(dbTest);
% Test sonucu
yyTest = netOpt(xTest);
[YY,sinifYY] = ysaCikis2Sinif(yyTest,Siniflar);
[KiyasSonuc,HataYuzde] = modelDegerlendirme(SinifTest,sinifYY);

%% SONUCUN YAZILMASI

for k=1:length(SinifTest)
    Dosya=dbTest.Files{k};
    p=strfind(Dosya,'\');
    if isempty(p), error('Dosya adresinde \ işareti yok!'); end
    Dosya=dbTest.Files{k}(p(end)+1:end);
    fprintf('%02d  %24s %16s %14s %6d\n', k, Dosya,SinifTest{k,1},sinifYY{k,1},KiyasSonuc(k,1));
end
% Genel sonucun yazdırılması
fprintf('\n');
fprintf('Test dosyaları için Başarı Oranı(%%): %.2f [%d/%d] \n\n', 100-HataYuzde, sum(KiyasSonuc),numel(KiyasSonuc));
save("voiceRecognition.mat")

function [Siniflar,SinifEgitim,SinifTest]=siniflarinAtanmasi(dbEgitim,dbTest)
    % Eğitim ve Test Sınıfları 
    SinifEgitim=cellstr(string(dbEgitim.Labels)); 
    SinifTest=cellstr(string(dbTest.Labels)); 
    Siniflar=unique(SinifEgitim);

    if ~all(contains(Siniflar,unique(SinifEgitim))), error('Eğitim veritabanı genel Sınıf elemanlarından birini içeriyor !'); end
    if ~all(contains(Siniflar,unique(SinifTest))), error('Test veritabanı genel Sınıf elemanlarından birini içeriyor !'); end
end

function X = Dosya2YSAGirisData(db)
    % Verinin okunması
    Data=cell(1,length(db.Files));
    VeriBoyut=nan(2,length(db.Files));
    for k=1:length(db.Files)
        Data{1,k}=audioread(db.Files{k});
        [VeriBoyut(1,k),VeriBoyut(2,k)]=size(Data{1,k});
    end
    % Veri Uzunluklarının kontrol edilmesi
    if all(VeriBoyut(1,:)==VeriBoyut(1,1)) && all(VeriBoyut(2,:)==VeriBoyut(2,1)) 
        X=nan(max(VeriBoyut(:,1)),length(Data));
        for k=1:size(X,2)
            X(:,k)=Data{1,k};
        end
    end   
end

function Y = Sinif2YSACikisData(Siniflar,SinifEgitim)
    Y = zeros(length(Siniflar), size(SinifEgitim,1));
    for n = 1: length(Siniflar)
        mask=strcmpi(SinifEgitim,Siniflar{n});
        Y(n,:) = double(mask');
    end
end

function  [YY,sinifYY] = ysaCikis2Sinif(yy,Siniflar)
    YY=zeros(size(yy));
    sinifYY=cell(size(yy,2),1);
    for k=1:size(yy,2)
        % YY çıktısının atanması
        [~,p]=max(yy(:,k));
        YY(p,k)=1;     
        % YY Sınıf bilgisinin atanması
        sinifYY{k,1}=Siniflar{p,1};
    end
end

function [KiyasSonuc,HataYuzde] = modelDegerlendirme(GozlemCikti,ModelCikti)
    if ~isequal(size(GozlemCikti),size(ModelCikti)), error('Gözlem ve Model veri boyutları aynı değil!'); end  
    % Kıyas sounucunun atanması
    KiyasSonuc=zeros(size(GozlemCikti));
    for k=1:length(GozlemCikti)
        if strcmpi(GozlemCikti{k,1},ModelCikti{k,1})
            KiyasSonuc(k,1)=1;
        end
    end   
    % Hata yüzdesinin atanması
    HataYuzde = 100-100.*(sum(KiyasSonuc)/numel(KiyasSonuc));
end