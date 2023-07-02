# Prediksi_Haraga_Saham_Bank_Mandiri

1. DOMAIN PROYEK

Prediksi nilai saham adalah topik menarik dalam bidang pembelajaran mesin. Hal ini karena faktor-faktor yang memengaruhi harga saham tidak dapat diprediksi dengan pasti[1]. Dengan mengamati dan menganalisis permintaan dan penawaran saham, kita dapat menentukan arah pergerakan harga saham. Jika permintaan melebihi penawaran, harga saham cenderung naik, sedangkan jika permintaan lebih rendah dari penawaran, harga saham cenderung turun[2]. Ada dua jenis analisis yang umum digunakan dalam investasi saham, yaitu analisis fundamental dan teknikal. Analisis fundamental melibatkan evaluasi kondisi perusahaan baik di masa lalu maupun perkiraan di masa depan untuk memprediksi harga saham, dengan menggunakan informasi dari berita dan laporan keuangan perusahaan. Sementara itu, analisis teknikal dilakukan berdasarkan pergerakan harga saham di periode sebelumnya[3].

Penggunaan pembelajaran mesin (Machine Learning) adalah metode yang akurat dalam memprediksi harga saham. Pembelajaran mesin merujuk pada kemampuan komputer atau program untuk memproses data dan digunakan sebagai sumber informasi dalam pengambilan keputusan dan pemecahan masalah[4]. Saham dapat dikategorikan sebagai deret waktu yang tidak stabil, dan telah dikembangkan beberapa teknik untuk memprediksi pergerakan harga saham. Salah satu teknik yang banyak digunakan oleh investor adalah Support Vector Machine (SVM) untuk memprediksi pergerakan indeks harga saham[5].

2. BUSINESS UNDERSTANDING

    2.1. Problem Statement 

Dalam konteks ini, tujuan investor adalah membuat keputusan investasi yang cerdas dan terinformasi mengenai saham PT. Bank Mandiri. Investor berharap dapat menggunakan prediksi harga saham sebagai panduan untuk mengambil keputusan investasi yang tepat waktu dan menguntungkan[2]. Oleh karena itu, analisis indeks harga saham PT. Bank Mandiri menjadi penting untuk memperoleh pemahaman yang mendalam tentang tren dan perilaku pasar.

Contoh Kasus:
Untuk menggambarkan bagaimana pemanfaatan algoritma SVM (Support Vector Machine) dapat membantu investor dalam mengambil keputusan investasi pada PT. Bank Mandiri, berikut ini merupakan contoh kasusnya:

Seorang investor yang berminat untuk membeli saham PT. Bank Mandiri ingin mengetahui apakah harga saham perusahaan tersebut akan mengalami kenaikan atau penurunan dalam periode waktu tertentu. Untuk mencapai tujuan ini, investor melakukan pengumpulan data historis mengenai harga saham Bank Mandiri seperti harga open, high, low, closing, dan volume.

Setelah mengumpulkan data tersebut, investor menggunakan algoritma SVM untuk melakukan analisis terhadap indeks harga saham PT. Bank Mandiri. Algoritma SVM dipilih karena reputasinya yang terbukti dalam pemodelan dan prediksi tren pasar yang kompleks.

Investor memisahkan data menjadi dua bagian: data latihan (untuk melatih model SVM) dan data pengujian (untuk menguji keakuratan model). Data latihan berisi informasi historis mengenai harga saham Bank Mandiri, sedangkan data pengujian berisi data harga saham yang lebih baru dan belum dikenal oleh model.

Setelah melatih model SVM menggunakan data latihan, investor dapat memanfaatkan model tersebut untuk memprediksi harga saham PT. Bank Mandiri dalam periode waktu tertentu berdasarkan data pengujian. Model SVM akan memberikan prediksi apakah harga saham Bank Mandiri diperkirakan akan mengalami kenaikan atau penurunan.

Investor kemudian dapat menggunakan prediksi harga saham sebagai salah satu faktor dalam pengambilan keputusan investasi. Sebagai contoh, apabila model SVM menunjukkan bahwa harga saham Bank Mandiri diperkirakan akan mengalami kenaikan, investor mungkin akan mempertimbangkan untuk membeli saham tersebut. Di sisi lain, jika prediksi menunjukkan penurunan harga saham, investor dapat memutuskan untuk menjual atau menunda pembelian saham.

Dengan memanfaatkan algoritma SVM dan melakukan analisis terhadap indeks harga saham PT. Bank Mandiri, investor dapat memperoleh wawasan yang lebih baik mengenai tren pasar dan membuat keputusan investasi yang lebih terinformasi. Meskipun demikian, perlu diingat bahwa prediksi pasar saham tidak selalu akurat secara mutlak, sehingga keputusan investasi harus didasarkan pada informasi yang komprehensif dan pemahaman yang mendalam mengenai pasar secara keseluruhan.

    2.2 Goals(tujuan)

Tujuan dari analisis indeks harga saham PT. Bank Mandiri menggunakan algoritma SVM adalah sebagai berikut:
Membantu investor dalam mengidentifikasi peluang investasi yang menguntungkan pada saham PT. Bank Mandiri.
Memperoleh pemahaman yang lebih baik tentang tren dan perilaku pasar saham Bank Mandiri.
Mengoptimalkan keputusan investasi dengan memanfaatkan prediksi harga saham yang dihasilkan oleh model SVM.
Memaksimalkan keuntungan investor melalui pembelian dan penjualan saham pada waktu yang tepat.
Mengurangi risiko investasi dengan memperoleh wawasan yang lebih terinformasi mengenai pergerakan harga saham Bank Mandiri.
Dengan mencapai tujuan-tujuan ini, investor dapat meningkatkan kinerja investasi mereka dan mencapai hasil yang lebih baik dalam pasar saham PT. Bank Mandiri.

3. DATA UNDERSTANDING

Data pada project ini menggunakan dataset dari Kaggle.com (https://www.kaggle.com/datasets/muamkh/ihsgstockdata) 

    3.1 Overview Data
- Nama dataset : Data stock daily diambil dari tanggal 16 April 2001 sampai dengan 6 Januari 2023.

Jadi, berdasarkan informasi data daily harga saham bank mandiri yaitu 5085 data dengan struktur data saham (variabel) yaitu: timestamp, open, low, high, close, dan volume.
Ulasan Variabel:
- timestamp = Date and time of stock transaction
- open = opening price
- low = lowest price in the timespan
- high = highest price in the timespan
- close = closing price
- volume = Total volume traded in the timespan

Gambar 1. 

<img width="283" alt="4" src="https://github.com/olan24/Prediksi_Saham/assets/68806443/cba019d3-57c3-478e-9d75-ca80347f008e">

Jadi, gambaran tentang struktur dan jenis data yang terdapat dalam DataFrame. Hal ini penting dalam analisis data lebih lanjut, seperti visualisasi, pemodelan, atau penarikan kesimpulan berdasarkan dataset tersebut.
Hasil perhitungan di atas adalah statistik ringkasan (summary statistics) yang diberikan untuk setiap kolom dalam data. Berikut adalah penjelasan detail untuk setiap statistik yang diberikan:

1. Kolom "open":
   - Count: Jumlah data yang tersedia dalam kolom "open" adalah 5085.
   - Mean: Rata-rata dari data dalam kolom "open" adalah 4035.6875123.
   - Standard Deviation (Std): Standar deviasi dari data dalam kolom "open" adalah 2522.9136979.
   - Minimum (Min): Nilai terendah dalam kolom "open" adalah 368.
   - 25th Percentile (25%): Nilai persentil ke-25 dari data dalam kolom "open" adalah 1450.
   - Median (50%): Nilai tengah (median) dari data dalam kolom "open" adalah 4125.
   - 75th Percentile (75%): Nilai persentil ke-75 dari data dalam kolom "open" adalah 6025.
   - Maximum (Max): Nilai tertinggi dalam kolom "open" adalah 10875.

2. Kolom "low":
   - Statistik yang diberikan mirip dengan kolom "open", tetapi berlaku untuk data dalam kolom "low".

3. Kolom "high":
   - Statistik yang diberikan mirip dengan kolom "open", tetapi berlaku untuk data dalam kolom "high".

4. Kolom "close":
   - Statistik yang diberikan mirip dengan kolom "open", tetapi berlaku untuk data dalam kolom "close".

5. Kolom "volume":
   - Statistik yang diberikan mirip dengan kolom "open", tetapi berlaku untuk data dalam kolom "volume".

Pada umumnya, statistik ringkasan digunakan untuk memberikan gambaran singkat tentang distribusi data dalam suatu kolom. Count menunjukkan jumlah data yang tersedia, mean memberikan informasi tentang nilai rata-rata, std mengukur dispersi atau variasi data, min dan max menunjukkan rentang nilai, dan persentil memberikan informasi tentang distribusi data dalam kuartil tertentu (25%, 50%, 75%). Statistik ini dapat membantu dalam analisis dan pemahaman data yang ada. tampilan penjelasan dapat di uraikan pada gambar 2:

Gambar 2. 

<img width="384" alt="2" src="https://github.com/olan24/Prediksi_Saham/assets/68806443/6e82ce3c-b33b-481e-8d0f-c4cab9454288">

4. DATA PREPARATION

a. Untuk mengetahui sebaran distribusi data kecenderungan pusat, serta adanya nilai ekstrem atau outlier dalam setiap fitur maka perlu dibuatkan sebuah plot sebagai gambaran. tahap ini dilakukan pengelompokkan data harian berdasarkan tahun dan menghitung rata-rata nilai pada kolom-kolom 'open', 'high', 'low', dan 'close'. Selanjutnya, membuat subplot dengan ukuran (20, 10) untuk menampilkan 4 grafik bar terpisah, masing-masing untuk kolom-kolom tersebut.

Gambar 3. plot 4 grafik

<img width="756" alt="5" src="https://github.com/olan24/Prediksi_Saham/assets/68806443/95b3ada4-e7a8-4bcb-a20c-f45bba88a614">

jadi, Berdasarkan Nilai kuartal akhir Bank Mandiri setelah melakukan pengumuman kuartal akhir mengalami penurunan pada harga close. Begitupun juga yang terjadi pada volume turut mengalami penurunan.

Gambar 4. 

![1](https://github.com/olan24/Prediksi_Saham/assets/68806443/3bcb462f-a120-4261-90b9-bfb8bc41d517)

dari matrix correlation diatas, dapat diketahui:

- Korelasi antara timestamp dan year sangat tinggi (0.998682). Ini menunjukkan bahwa fitur timestamp dan year memiliki korelasi yang kuat dan hampir identik. Karena itu, mungkin ada redundansi informasi antara kedua fitur tersebut.

- Korelasi antara timestamp dan low-high adalah -0.646914, yang menunjukkan adanya korelasi negatif yang kuat antara timestamp dan perbedaan nilai low-high. Hal ini bisa menunjukkan bahwa dalam periode tertentu, jika timestamp semakin tinggi, perbedaan nilai antara low dan high semakin rendah.

- Korelasi antara month dan is_quarter_end adalah 0.201347. Ini menunjukkan adanya korelasi positif yang sedang antara bulan dan penanda akhir kuartal. Artinya, kemungkinan besar penanda akhir kuartal muncul pada bulan-bulan tertentu.

- Korelasi antara open-close dan target adalah 0.008508. Ini menunjukkan adanya korelasi positif yang sangat lemah antara perbedaan nilai open-close dan target. Hal ini menunjukkan bahwa perbedaan nilai antara open dan close mungkin memiliki pengaruh minimal terhadap nilai target.

- Korelasi antara low-high dan target adalah -0.038303. Ini menunjukkan adanya korelasi negatif yang lemah antara perbedaan nilai low-high dan target. Hal ini menunjukkan bahwa perbedaan nilai antara low dan high juga memiliki pengaruh minimal terhadap nilai target.

Tabel 1. Correlation matrix


timestamp      close     volume        day      month    \textbackslash{}         &  &  &  &  &  &  \\


timestamp       1.0000000  0.9506830 -0.2801296 -0.0011882  0.0102865             &  &  &  &  &  &  \\

close           0.9506830  1.0000000 -0.2875390 -0.0027040  0.0066413             &  &  &  &  &  &  \\

volume         -0.2801296 -0.2875390  1.0000000 -0.0082141 -0.0714209             &  &  &  &  &  &  \\

day            -0.0011882 -0.0027040   -0.0082141  1.0000000  0.0136003           &  &  &  &  &  &  \\

month           0.0102865  0.0066413 -0.0714209  0.0136003    1.0000000           &  &  &  &  &  &  \\

year            0.9986815  0.9496046 -0.2761936 -0.0061524   -0.0408802           &  &  &  &  &  &  \\

is\_quarter\_end  0.0055890 -0.0017293 -0.0584188  0.0020466    0.2013467         &  &  &  &  &  &  \\

open-close      0.0256339 -0.0006410 -0.0549612   -0.0020995 -0.0245449           &  &  &  &  &  &  \\


low-high       -0.6469144 -0.6134978 -0.0301265  0.0088819 -0.0071288             &  &  &  &  &  &  \\

target          0.0475509  0.0309021    0.0254940 -0.0002886 -0.0122267           &  &  &  &  &  &  \\
                                                                                  &  &  &  &  &  &  \\


year  is\_quarter\_end  open-close     low-high     target                        &  &  &  &  &  &  \\

timestamp       0.9986815       0.0055890   0.0256339 -0.6469144  0.0475509       &  &  &  &  &  &  \\

close           0.9496046      -0.0017293  -0.0006410 -0.6134978  0.0309021       &  &  &  &  &  &  \\

volume         -0.2761936      -0.0584188  -0.0549612 -0.0301265  0.0254940       &  &  &  &  &  &  \\

day            -0.0061524       0.0020466  -0.0020995    0.0088819 -0.0002886     &  &  &  &  &  &  \\

month          -0.0408802       0.2013467  -0.0245449 -0.0071288 -0.0122267       &  &  &  &  &  &  \\

year            1.0000000      -0.0046678   0.0268740 -0.6460770  0.0481405       &  &  &  &  &  &  \\

is\_quarter\_end   -0.0046678       1.0000000  -0.0013658    0.0326007  0.0001637 &  &  &  &  &  &  \\

open-close      0.0268740      -0.0013658   1.0000000    0.0514266  0.0085083     &  &  &  &  &  &  \\

low-high       -0.6460770       0.0326007   0.0514266    1.0000000 -0.0383031     &  &  &  &  &  &  \\

target          0.0481405       0.0001637   0.0085083 -0.0383031  1.0000000       &  &  &  &  &  & 

\end{tabular}
\end{table}


b. Teknik preparation yang digunakan adalah standart scaler. StandardScaler adalah salah satu transformer yang digunakan dalam pemrosesan data serta dalam analisis data dan pemodelan statistik. StandardScaler digunakan untuk menormalkan atau menskalakan fitur-fitur numerik dalam sebuah dataset. Pemrosesan ini melakukan penskalaan fitur-fitur dengan menghilangkan rata-rata dan menskalakan varians menjadi 1.

c. Split Data: Pada tahapan ini data dibagi meliputi Data Train 4576 (90%) dan Data Valid/Test 509 (10%) dari keseluruhan data.


5. MODELLING

Pada tugas ini, penulis menggunakan algoritma Support Vector Regression (SVR) untuk melakukan regresi dengan kernel linear. Berikut adalah penjelasan tahapan dan parameter yang digunakan:

1. Membuat objek regressor SVR:
   `regressor = SVR(kernel='linear', C=1.0, epsilon=0.1)`
   - `SVR` adalah kelas regressor dari modul `sklearn.svm` yang digunakan untuk melakukan regresi dengan Support Vector Machines (SVM).
   - `kernel='linear'` menentukan bahwa kita menggunakan kernel linier untuk model SVR. Kernel linier mengasumsikan bahwa hubungan antara fitur dan target adalah linear.
   - `C=1.0` adalah parameter yang mengontrol toleransi terhadap kesalahan dalam model. Semakin tinggi nilai C, semakin ketat model akan menyesuaikan data pelatihan.
   - `epsilon=0.1` adalah lebar jendela toleransi kesalahan. Nilai ini mengontrol seberapa banyak poin data pelatihan dapat jatuh di luar batas toleransi.

2. Melatih model dengan data pelatihan:
   `regressor.fit(x_train, y_train)`
   - `x_train` adalah matriks fitur dari data pelatihan yang digunakan untuk melatih model.
   - `y_train` adalah vektor target dari data pelatihan yang digunakan untuk melatih model.

3. Memprediksi target menggunakan data validasi:
   `y_pred = regressor.predict(x_valid)`
   - `x_valid` adalah matriks fitur dari data validasi yang digunakan untuk melakukan prediksi.
   - `y_pred` adalah vektor hasil prediksi target untuk data validasi.

Dalam rangkaian ini, SVR digunakan untuk mempelajari hubungan linier antara fitur dan target. Parameter C dan epsilon digunakan untuk mengontrol kompleksitas model dan toleransi terhadap kesalahan. Setelah melatih model, kita menggunakan model yang terlatih untuk memprediksi target pada data validasi.

6. EVALUASI/RESULT

- Kesalahan Kuadrat Rata-Rata (MSE):
MSE adalah ukuran yang memperkirakan kesalahan kuadrat rata-rata antara nilai prediksi dan nilai sebenarnya dari kumpulan data. Semakin rendah nilai MSE, semakin baik model regresi memprediksi nilai target.
MSE dihitung dengan menjumlahkan kesalahan kuadrat (selisih antara nilai prediksi dan nilai aktual) untuk setiap titik data, lalu membaginya dengan jumlah total titik data. Rumusnya adalah sebagai berikut:

MSE = (1/n) * Σ(yi - ŷi)^2

di mana yi adalah nilai sebenarnya, ŷi adalah nilai prediksi, dan n adalah jumlah titik data.

- R-kuadrat (R2):
R2 adalah metrik evaluasi yang mengukur sejauh mana variasi variabel target dapat dijelaskan oleh model regresi. Nilai R2 berkisar dari 0 hingga 1, dengan 0 menunjukkan bahwa model tidak dapat menjelaskan variasi sama sekali dan 1 menunjukkan bahwa model dapat menjelaskan semua variasi dengan sempurna.
R2 dihitung dengan membandingkan varian prediksi model dengan varian nilai target aktual. Nilai R2 yang positif menunjukkan bahwa model membuat prediksi yang lebih baik daripada prediksi yang hanya menggunakan rata-rata. Rumusnya adalah sebagai berikut:

R2 = 1 - (SSR/SST)

di mana SSR adalah jumlah residu kuadrat dan SST adalah jumlah total kuadrat.
Secara umum, semakin tinggi nilai R2, semakin baik model regresi dalam menjelaskan variasi data. Namun, R2 juga memiliki keterbatasan dan harus dianalisis bersama dengan metrik lainnya. 

- RMSE 
RMSE adalah akar kuadrat dari MSE. RMSE cenderung memungkinkan interpretasi yang lebih intuitif karena memiliki unit yang sama dengan variabel target asli. Seperti halnya MSE, semakin rendah nilai RMSE, semakin baik model regresi memprediksi nilai target.
Rumus RMSE:

RMSE = √MSE 
Pada hasil evaluasi modelling data ini menunjukkan MSE, MAE, R-Square (R2), MedAE, dan RMSE pada algoritma SVR adalah: 
- Mean squared error(MSE) =  0.34;
- Mean absolute error(MAE) =  0.43;
- Median absolute error(MedAE) =  0.1;
- RMSE: 0.5848034889023938;
- R^2: 0.912;

7. KESIMPULAN

Dalam penelitian ini dapat disimpulkan sebagai berikut:

Dalam penelitian ini dapat disimpulkan sebagai berikut:

Dari hasil evaluasi yang dilakukan, dapat disimpulkan bahwa model prediksi memiliki kualitas yang baik dan mampu memberikan hasil yang cukup akurat. Berikut adalah analisis lebih lanjut terkait beberapa nilai evaluasi yang diperoleh:

- Mean squared error (MSE) sebesar 0.34 menunjukkan bahwa rata-rata selisih kuadrat antara nilai aktual dan nilai prediksi adalah relatif kecil. Ini menunjukkan tingkat kesalahan yang rendah dalam model prediksi. Mean absolute error (MAE) sebesar 0.43 menunjukkan bahwa rata-rata selisih mutlak antara nilai aktual dan nilai prediksi juga rendah. Artinya, prediksi cenderung mendekati nilai aktual dengan baik.

- Median absolute error (MedAE) sebesar 0.1 mengindikasikan bahwa sebagian besar kesalahan prediksi terletak pada rentang yang relatif kecil. Hal ini menunjukkan konsistensi model dalam memprediksi data dengan tingkat kesalahan yang rendah.

- Root Mean Square Error (RMSE) sebesar 0.5848 menunjukkan akurasi yang tinggi dalam memprediksi nilai aktual. RMSE menggambarkan sejauh mana perbedaan antara nilai aktual dan nilai prediksi secara keseluruhan. Semakin kecil nilai RMSE, semakin akurat prediksi model.

- R-squared (R^2) sebesar 0.912 menandakan bahwa model mampu menjelaskan 91.2% variasi dalam data aktual. Semakin tinggi nilai R^2, semakin baik model dapat menggambarkan pola dan tren yang ada dalam data aktual.

Berdasarkan kesimpulan di atas, dapat dikatakan bahwa model prediksi yang dievaluasi memiliki performa yang baik. Meskipun tidak ada model yang sempurna, nilai-nilai evaluasi yang rendah, seperti MSE, MAE, MedAE, RMSE, dan tingkat keakuratan yang tinggi dengan R^2, menunjukkan bahwa model cenderung memberikan hasil prediksi yang akurat dan konsisten. Namun, tetap perlu dilakukan analisis lebih lanjut untuk memastikan validitas model dan memperhatikan konteks dan tujuan prediksi yang ingin dicapai.


8. SARAN

- Evaluasi metrik: Selain MSE, R-squared, dan RMSE, ada baiknya juga melihat metrik evaluasi lainnya seperti MAE (Mean Absolute Error) dan MAPE (Mean Absolute Percentage Error) untuk mendapatkan gambaran yang lebih komprehensif tentang performa model. Metrik evaluasi tambahan ini dapat memberikan wawasan yang lebih lengkap tentang kesalahan model dalam memprediksi nilai target.

- Penyempurnaan model: Mengingat performa yang rendah dalam menjelaskan variasi dalam data, disarankan untuk melakukan penyempurnaan pada model-model yang digunakan. Mungkin perlu mengubah konfigurasi atau parameter pada algoritma KNN, RF, dan NN untuk meningkatkan performa prediksi. Eksperimen dengan berbagai parameter dan teknik tuning dapat membantu meningkatkan hasil prediksi.

- Pemilihan fitur: Perlu dipertimbangkan pemilihan fitur yang lebih relevan dan informatif untuk meningkatkan performa model. Evaluasi ulang fitur yang digunakan dalam model dapat membantu dalam memilih fitur yang lebih penting dan memiliki hubungan yang lebih kuat dengan variabel target.

- Data tambahan: Jika memungkinkan, penambahan data tambahan atau pengumpulan data yang lebih lengkap dan representatif dapat meningkatkan performa model. Dengan memiliki lebih banyak data, model dapat menemukan pola yang lebih baik dan memberikan prediksi yang lebih akurat.

- Model alternatif: Selain algoritma yang telah digunakan, ada baiknya juga mengevaluasi model alternatif. Mungkin ada algoritma lain yang lebih cocok atau memiliki performa yang lebih baik dalam menyelesaikan masalah ini. Mengeksplorasi model lain seperti regresi linear, decision tree, atau ensemble model lainnya dapat memberikan pemahaman yang lebih baik tentang mana model yang paling sesuai untuk dataset ini.

- Validasi ulang: Melakukan validasi ulang terhadap model yang telah ditingkatkan dan melakukan perbandingan dengan model-model alternatif. Validasi silang (cross-validation) atau penggunaan dataset validasi yang lebih besar dapat memberikan kepercayaan yang lebih tinggi terhadap performa model.

Dengan melakukan penyempurnaan pada pemodelan, pemilihan fitur yang tepat, penambahan data, dan evaluasi model alternatif, diharapkan dapat mencapai hasil yang lebih baik dalam memprediksi nilai target dan meningkatkan kemampuan model dalam menjelaskan variasi dalam data.

9. DAFTAR ISI

[1] K. M. Hindrayani, I. G. S. Mas Diyasa, P. A. Riyantoko, dan T. M. Fahrudin, “Studi Literatur Mengenai Prediksi Harga Saham Menggunakan Machine Learning,” Pros. Semin. Nas. Inform. Bela Negara, vol. 1, hal. 71–75, 2020, doi: 10.33005/santika.v1i0.20.

[2] R. H. Kusumodestoni dan S. Sarwido, “Komparasi Model Support Vector Machines (Svm) Dan Neural Network Untuk Mengetahui Tingkat Akurasi Prediksi Tertinggi Harga Saham,” J. Inform. Upgris, vol. 3, no. 1, 2017, doi: 10.26877/jiu.v3i1.1536.

[3] A. B. Untoro, “Prediksi Harga Saham Dengan Menggunakan Jaringan Syaraf Tiruan”, Jurnal Teknologi Informatika dan Komputer MH Thamrin, vol. 6, no. 2, pp.103-111, 2020.

[4] Patriya, “Implementasi Support Vector Machine Pada Prediksi Harga Saham Gabungan (IHSG),” Jurnal Ilmiah Teknologi dan Rekayasa, vol. 25, no. 1, pp. 24–38, 2020.

[5] F. R. Setiawan, R. F. Umbara, and A.A. Rohmawati, “Prediksi Pergerakan Harga Saham dengan Metode Support Vector Machine (SVM) Menggunakan Trend Deterministic Data Preparation”, e-Proxeeding of Engineering, vol. 5m no. 3, pp.8356-8372, 2018.

[6] R. Akmalia, I. Slamet, dan H. Pratiwi, “Analisis Sentimen Twitter Berbahasa Indonesia Terhadap Aplikasi,” Pros. Semin. Nas. MIPA UNIPA, vol. 2022, hal. 150–156, 2022.

[7] R. Primartha, Algoritma Machine Learning. Informatika, 2021.

[8] R. T. Handayanto dan Herlawati, Data Mining dan Machine Learning Menggunakan Matlab & Python. Penerbit informatika, 2020.

