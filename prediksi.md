# LAPORAN PROYEK MACHINE LEARNING - ANDY OLAN SITORUS

## DOMAIN PROYEK

Prediksi nilai saham adalah topik menarik dalam bidang pembelajaran mesin. Hal ini karena faktor-faktor yang memengaruhi harga saham tidak dapat diprediksi dengan pasti[1]. Dengan mengamati dan menganalisis permintaan dan penawaran saham, kita dapat menentukan arah pergerakan harga saham. Jika permintaan melebihi penawaran, harga saham cenderung naik, sedangkan jika permintaan lebih rendah dari penawaran, harga saham cenderung turun[2]. Ada dua jenis analisis yang umum digunakan dalam investasi saham, yaitu analisis fundamental dan teknikal. Analisis fundamental melibatkan evaluasi kondisi perusahaan baik di masa lalu maupun perkiraan di masa depan untuk memprediksi harga saham, dengan menggunakan informasi dari berita dan laporan keuangan perusahaan. Sementara itu, analisis teknikal dilakukan berdasarkan pergerakan harga saham di periode sebelumnya[3].

Penggunaan pembelajaran mesin adalah metode yang akurat dalam memprediksi harga saham. Pembelajaran mesin merujuk pada kemampuan komputer atau program untuk memproses data dan digunakan sebagai sumber informasi dalam pengambilan keputusan dan pemecahan masalah[4]. Saham dapat dikategorikan sebagai deret waktu yang tidak stabil, dan telah dikembangkan beberapa teknik untuk memprediksi pergerakan harga saham. Salah satu teknik yang banyak digunakan oleh investor adalah Support Vector Machine (SVM) untuk memprediksi pergerakan indeks harga saham[5].


## BUSINESS UNDERSTANDING

### Problem Statement 

Dalam konteks ini, tujuan investor adalah membuat keputusan investasi yang cerdas dan terinformasi mengenai saham PT. Bank Mandiri. Investor berharap dapat menggunakan prediksi harga saham sebagai panduan untuk mengambil keputusan investasi yang tepat waktu dan menguntungkan[2]. Oleh karena itu, analisis indeks harga saham PT. Bank Mandiri menjadi penting untuk memperoleh pemahaman yang mendalam tentang tren dan perilaku pasar.

Contoh Kasus:
Untuk menggambarkan bagaimana pemanfaatan algoritma SVM (Support Vector Machine) dapat membantu investor dalam mengambil keputusan investasi pada PT. Bank Mandiri, berikut ini merupakan contoh kasusnya:

Seorang investor yang berminat untuk membeli saham PT. Bank Mandiri ingin mengetahui apakah harga saham perusahaan tersebut akan mengalami kenaikan atau penurunan dalam periode waktu tertentu. Untuk mencapai tujuan ini, investor melakukan pengumpulan data historis mengenai harga saham Bank Mandiri seperti harga open, high, low, closing, dan volume.

Setelah mengumpulkan data tersebut, investor menggunakan algoritma SVM untuk melakukan analisis terhadap indeks harga saham PT. Bank Mandiri. Algoritma SVM dipilih karena reputasinya yang terbukti dalam pemodelan dan prediksi tren pasar yang kompleks.

Investor memisahkan data menjadi dua bagian: data latihan (untuk melatih model SVM) dan data pengujian (untuk menguji keakuratan model). Data latihan berisi informasi historis mengenai harga saham Bank Mandiri, sedangkan data pengujian berisi data harga saham yang lebih baru dan belum dikenal oleh model.

Setelah melatih model SVM menggunakan data latihan, investor dapat memanfaatkan model tersebut untuk memprediksi harga saham PT. Bank Mandiri dalam periode waktu tertentu berdasarkan data pengujian. Model SVM akan memberikan prediksi apakah harga saham Bank Mandiri diperkirakan akan mengalami kenaikan atau penurunan.

Investor kemudian dapat menggunakan prediksi harga saham sebagai salah satu faktor dalam pengambilan keputusan investasi. Sebagai contoh, apabila model SVM menunjukkan bahwa harga saham Bank Mandiri diperkirakan akan mengalami kenaikan, investor mungkin akan mempertimbangkan untuk membeli saham tersebut. Di sisi lain, jika prediksi menunjukkan penurunan harga saham, investor dapat memutuskan untuk menjual atau menunda pembelian saham.

Dengan memanfaatkan algoritma SVM dan melakukan analisis terhadap indeks harga saham PT. Bank Mandiri, investor dapat memperoleh wawasan yang lebih baik mengenai tren pasar dan membuat keputusan investasi yang lebih terinformasi. Meskipun demikian, perlu diingat bahwa prediksi pasar saham tidak selalu akurat secara mutlak, sehingga keputusan investasi harus didasarkan pada informasi yang komprehensif dan pemahaman yang mendalam mengenai pasar secara keseluruhan.

berdasarkan penjelasan di atas, maka problem statement yang akan di angkat adalah:
* Bagaimana hasil prediksi MSE, MAE, R-Square (R2), MedAE, dan RMSE pada harga saham PT. Bank Mandiri menggunakan algoritma Support Vector Machine?

### Goals(tujuan)

Tujuan dari tugas/proyek ini adalah mengetahui hasil prediksi MSE, MAE, R-Square (R2), MedAE, dan RMSE pada harga saham PT. Bank Mandiri menggunakan algoritma Support Vector Machine.

Dengan mencapai tujuan ini, investor dapat meningkatkan kinerja investasi mereka dan mencapai hasil yang lebih baik dalam pasar saham PT. Bank Mandiri.


## DATA UNDERSTANDING

Data pada project ini menggunakan dataset dari Kaggle.com (https://www.kaggle.com/datasets/muamkh/ihsgstockdata) 

### Overview Data
- Nama dataset : Data stock daily diambil dari tanggal 16 April 2001 sampai dengan 6 Januari 2023.

Jadi, berdasarkan informasi data daily harga saham bank mandiri yaitu 5085 data dengan struktur data saham (variabel) yaitu: timestamp, open, low, high, close, dan volume.
Ulasan Variabel:
- timestamp = Date and time of stock transaction
- open = opening price
- low = lowest price in the timespan
- high = highest price in the timespan
- close = closing price
- volume = Total volume traded in the timespan

Tabel 1. Data Info

| Timestamp  | open | low | high | volume     | close |
|------------|------|-----|------|------------|-------|
| 2003-07-14 | 381  | 381 | 430  | 1198338043 | 417   |
| 2003-07-15 | 430  | 417 | 442  | 420912767  | 417   |
| 2003-07-16 | 417  | 405 | 417  | 73507314   | 405   |
| 2003-07-17 | 405  | 405 | 417  | 111095912  | 417   |

Jadi, gambaran tentang struktur dan jenis data yang terdapat dalam DataFrame. Hal ini penting dalam analisis data lebih lanjut, seperti visualisasi, pemodelan, atau penarikan kesimpulan berdasarkan dataset tersebut. data yang di tampilkan adalah 4 baris pertama.

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

Tabel 2. tampilan Describe data

|       | open         | low          | high         | close        | volume          |
|-------|--------------|--------------|--------------|--------------|-----------------|
| Count | 5085.000000  | 5085.000000  | 5085.000000  | 5085.0000000 | 5.0850000000000 |
| mean  | 403.6875     | 3980.236578  | 4084.236578  | 4035.170305  | 6.1891860000000 |
| std   | 2522.913698  | 2495.786202  | 2547.857617  | 2521.770583  | 6.2297790000000 |
| min   | 368.000000   | 344.000000   | 368.000000   | 368.000000   | 0.0000000000000 |
| 25%   | 1450.000000  | 1425.000000  | 1487.000000  | 1450.000000  | 2.9246310000000 |
| 50%   | 4125.000000  | 4060.000000  | 4200.000000  | 4125.000000  | 4.6665800000000 |
| 75%   | 6025.000000  | 5950.000000  | 6100.000000  | 6012.000000  | 7.5746000000000 |
| Max   | 10875.000000 | 10725.000000 | 11000.000000 | 10900.000000 | 1.1983380000000 |

### Visualisasi Data

Gambar di bawah ini menunjukkan plot histogram dan kurva kepadatan untuk setiap fitur dalam dataset menggunakan seaborn. Masing-masing subplot menunjukkan distribusi data dari fitur yang bersangkutan. Visualisasi ini membantu untuk memahami karakteristik data, seperti distribusi nilai, kecenderungan, atau pola dalam fitur-fitur tersebut. Dengan menggunakan subplot, setiap plot fitur ditampilkan secara terpisah dalam satu gambar, memudahkan pembandingan dan analisis antara fitur-fitur yang berbeda.

Gambar 1. Plot (open)
![1a](https://github.com/olan24/Prediksi_Saham/assets/68806443/34c19a2c-7740-482c-ad77-24b93dabc444)

Gambar 2. Plot (High)
![1b](https://github.com/olan24/Prediksi_Saham/assets/68806443/8a538907-e973-4542-a75d-7b0c87569e82)

Gambar 3. Plot (Low)
![1c](https://github.com/olan24/Prediksi_Saham/assets/68806443/c385c139-fc5c-480a-a18a-4ca6683ef77d)

Gambar 4. Plot (close)
![1d](https://github.com/olan24/Prediksi_Saham/assets/68806443/acf595f8-077a-4856-8ae3-dd8c6aeb1966)

Gambar 5. Plot (Volume)
![1e](https://github.com/olan24/Prediksi_Saham/assets/68806443/88d58d89-7c62-4662-b6aa-d6bc34b40508)

selanjutnya, visualisasi data dibawah ini menggunakan camdlestick plotly, khususnya `plotly.graph_objects`, untuk membuat grafik candlestick chart (grafik lilin) berdasarkan data harga saham. Berikut adalah penjelasan mengenai kegunaan kode tersebut:

1. `import plotly.graph_objects as go`:
   - Mengimpor modul `graph_objects` dari library Plotly. Modul ini digunakan untuk membuat visualisasi grafik dengan elemen-elemen yang lebih kompleks.

2. `from datetime import datetime`:
   - Mengimpor modul `datetime` dari library Python. Modul ini digunakan untuk memanipulasi dan memformat objek waktu.

3. `fig = go.Figure(data=[go.Candlestick(x=data['timestamp'], open=data['open'], high=data['high'], low=data['low'], close=data['close'])])`:
   - Membuat objek `Figure` dari `go.Figure()`. Ini adalah kontainer untuk menggambarkan data pada plotly.
   - Pada bagian `data`, menggunakan `go.Candlestick()` untuk membuat objek grafik candlestick chart.
   - Parameter `x` diisi dengan data kolom 'timestamp', yaitu data waktu pada sumbu x.
   - Parameter `open` diisi dengan data kolom 'open', yaitu data harga pembukaan saham pada sumbu y.
   - Parameter `high` diisi dengan data kolom 'high', yaitu data harga tertinggi saham pada sumbu y.
   - Parameter `low` diisi dengan data kolom 'low', yaitu data harga terendah saham pada sumbu y.
   - Parameter `close` diisi dengan data kolom 'close', yaitu data harga penutupan saham pada sumbu y.

4. `fig.show()`:
   - Menampilkan grafik candlestick chart yang telah dibuat menggunakan `fig.show()`.
   - Grafik akan ditampilkan dalam jendela pop-up atau output yang relevan tergantung pada lingkungan pengembangan yang digunakan.

Dengan menggunakan kode tersebut, kita dapat membuat visualisasi candlestick chart yang berguna untuk menganalisis harga saham pada suatu periode. Grafik ini menampilkan informasi harga pembukaan, harga tertinggi, harga terendah, dan harga penutupan pada sumbu y, sementara sumbu x menunjukkan data waktu. Candlestick chart sangat berguna dalam analisis teknikal dan memvisualisasikan perubahan harga saham dari waktu ke waktu. berikut visualisasi candlestick yang terdapat pada gambar 6:

Gambar 6. candlestick Plot 

<img width="647" alt="3" src="https://github.com/olan24/Prediksi_Saham/assets/68806443/16202e73-6661-4ecf-9fa9-21bd824ec753">

## DATA PREPARATION

a. Untuk mengetahui sebaran distribusi data kecenderungan pusat, serta adanya nilai ekstrem atau outlier dalam setiap fitur maka perlu dibuatkan sebuah plot sebagai gambaran. tahap ini dilakukan pengelompokkan data harian berdasarkan tahun dan menghitung rata-rata nilai pada kolom-kolom 'open', 'high', 'low', dan 'close'. Selanjutnya, membuat subplot dengan ukuran (20, 10) untuk menampilkan 4 grafik bar terpisah, masing-masing untuk kolom-kolom tersebut.

Gambar 3. plot 4 grafik

<img width="756" alt="5" src="https://github.com/olan24/Prediksi_Saham/assets/68806443/95b3ada4-e7a8-4bcb-a20c-f45bba88a614">

selanjutnya, pada tahap ini dilakukan memodifikasi dataframe data dengan menambahkan kolom-kolom baru yang memberikan informasi tambahan tentang perbedaan harga antara pembukaan dan penutupan, perbedaan harga terendah dan tertinggi, serta menandai target berdasarkan pergerakan harga penutupan pada periode berikutnya. Informasi ini dapat berguna dalam analisis dan pemodelan harga saham.

Tabel 3. Modifikasi tabel dengan menetapkan close sebagai target.
| Timestamp  | open  | low  | high  | close | volume   | day | month | year | is_quarter_end | open-close | low-high | target |
|------------|-------|------|-------|-------|----------|-----|-------|------|----------------|------------|----------|--------|
| 2023-01-05 | 10050 | 9725 | 10050 | 9825  | 42472900 | 5   | 1     | 2023 | 0              | 225        | -325     | 0      |
| 2023-01-06 | 9725  | 9600 | 9800  | 9800  | 22048500 | 6   | 1     | 2023 | 0              | -75        | -200     | 0      |

data di atas merupakan data representase.

selain itu, perlu dilakukan tahapan Correlation matrix. mengapa? karena hal ini merupakan representasi tabel dari korelasi antara setiap pasangan variabel dalam suatu dataset. Setiap elemen matriks menunjukkan korelasi antara dua variabel, di mana nilai korelasi dapat berkisar dari -1 hingga +1. Nilai +1 menunjukkan korelasi positif sempurna, nilai -1 menunjukkan korelasi negatif sempurna, dan nilai 0 menunjukkan tidak adanya korelasi linier antara dua variabel.

Kegunaan dari kode di atas adalah untuk menghasilkan visualisasi heatmap (peta panas) dari correlation matrix menggunakan library seaborn (sns). Berikut adalah penjelasan kegunaan dari kode tersebut:

1. `correlation_matrix = data.corr()`:
   - Menghitung matriks korelasi dengan memanggil metode `corr()` pada dataframe `data`.
   - Metode ini menghasilkan matriks korelasi yang berisi koefisien korelasi antara setiap pasangan variabel dalam dataset.

2. `plt.figure(figsize=(10, 10))`:
   - Membuat figure (gambar) dengan ukuran 10x10 untuk menampung heatmap.

3. `sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="RdBu")`:
   - Menggunakan fungsi `heatmap()` dari seaborn untuk membuat heatmap dari matriks korelasi.
   - Parameter `correlation_matrix` adalah matriks korelasi yang ingin divisualisasikan.
   - Parameter `annot=True` digunakan untuk menampilkan nilai-nilai korelasi pada heatmap.
   - Parameter `fmt=".2f"` mengatur format angka pada heatmap menjadi dua angka desimal.
   - Parameter `cmap="RdBu"` menentukan skema warna untuk heatmap (merah untuk korelasi negatif dan biru untuk korelasi positif).

4. `plt.title("Correlation Matrix")`:
   - Menambahkan judul "Correlation Matrix" pada heatmap.

5. `plt.show()`:
   - Menampilkan heatmap yang telah dibuat.

Dengan menggunakan kode tersebut, kita dapat memvisualisasikan matriks korelasi antara variabel-variabel dalam dataset menggunakan heatmap. Heatmap membantu dalam mengevaluasi hubungan antara variabel-variabel, mengidentifikasi pola korelasi, dan memahami tingkat kekuatan serta arah hubungan antar variabel. Informasi ini dapat digunakan untuk analisis, pemodelan, atau pemilihan fitur dalam dataset.
jadi, Berdasarkan Nilai kuartal akhir Bank Mandiri setelah melakukan pengumuman kuartal akhir mengalami penurunan pada harga close. Begitupun juga yang terjadi pada volume turut mengalami penurunan.

Gambar 7. korelasi Matriks

![1](https://github.com/olan24/Prediksi_Saham/assets/68806443/3bcb462f-a120-4261-90b9-bfb8bc41d517)

dari matrix correlation diatas, dapat diketahui:

- Korelasi antara timestamp dan year sangat tinggi (0.998682). Ini menunjukkan bahwa fitur timestamp dan year memiliki korelasi yang kuat dan hampir identik. Karena itu, mungkin ada redundansi informasi antara kedua fitur tersebut.

- Korelasi antara timestamp dan low-high adalah -0.646914, yang menunjukkan adanya korelasi negatif yang kuat antara timestamp dan perbedaan nilai low-high. Hal ini bisa menunjukkan bahwa dalam periode tertentu, jika timestamp semakin tinggi, perbedaan nilai antara low dan high semakin rendah.

- Korelasi antara month dan is_quarter_end adalah 0.201347. Ini menunjukkan adanya korelasi positif yang sedang antara bulan dan penanda akhir kuartal. Artinya, kemungkinan besar penanda akhir kuartal muncul pada bulan-bulan tertentu.

- Korelasi antara open-close dan target adalah 0.008508. Ini menunjukkan adanya korelasi positif yang sangat lemah antara perbedaan nilai open-close dan target. Hal ini menunjukkan bahwa perbedaan nilai antara open dan close mungkin memiliki pengaruh minimal terhadap nilai target.

- Korelasi antara low-high dan target adalah -0.038303. Ini menunjukkan adanya korelasi negatif yang lemah antara perbedaan nilai low-high dan target. Hal ini menunjukkan bahwa perbedaan nilai antara low dan high juga memiliki pengaruh minimal terhadap nilai target.

b. Teknik preparation yang digunakan adalah standart scaler. StandardScaler adalah salah satu transformer yang digunakan dalam pemrosesan data serta dalam analisis data dan pemodelan statistik. StandardScaler digunakan untuk menormalkan atau menskalakan fitur-fitur numerik dalam sebuah dataset. Pemrosesan ini melakukan penskalaan fitur-fitur dengan menghilangkan rata-rata dan menskalakan varians menjadi 1.

c. Split Data: Pada tahapan ini data dibagi meliputi Data Train 4068 (80%) dan Data Valid/Test 1017 (20%) dari keseluruhan data.

## MODELLING

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

## EVALUASI/RESULT

Hasil dari kode telah di bangun, menunjukkan beberapa metrik evaluasi yang digunakan untuk mengukur kualitas prediksi dari model regresi. Berikut adalah penjelasan dari setiap metrik evaluasi yang diberikan:

1. Mean Squared Error (MSE):
   - MSE mengukur rata-rata dari kuadrat selisih antara nilai prediksi dan nilai sebenarnya.
   - Rumus MSE: MSE = (1/n) * Σ(y_pred - y_actual)^2, di mana y_pred adalah nilai prediksi dan y_actual adalah nilai sebenarnya, dan n adalah jumlah data.
   - Nilai MSE yang lebih rendah menunjukkan kesalahan prediksi yang lebih kecil dan kualitas prediksi yang lebih baik.

2. Mean Absolute Error (MAE):
   - MAE mengukur rata-rata dari selisih absolut antara nilai prediksi dan nilai sebenarnya.
   - Rumus MAE: MAE = (1/n) * Σ|y_pred - y_actual|, di mana y_pred adalah nilai prediksi dan y_actual adalah nilai sebenarnya, dan n adalah jumlah data.
   - Nilai MAE yang lebih rendah menunjukkan kesalahan prediksi yang lebih kecil dan kualitas prediksi yang lebih baik.

3. Median Absolute Error (MedAE):
   - MedAE mengukur nilai tengah dari selisih absolut antara nilai prediksi dan nilai sebenarnya.
   - Rumus MedAE: MedAE = Median(|y_pred - y_actual|), di mana y_pred adalah nilai prediksi dan y_actual adalah nilai sebenarnya.
   - MedAE mengabaikan nilai outlier dan memberikan gambaran tentang kesalahan prediksi yang lebih stabil.

4. Root Mean Squared Error (RMSE):
   - RMSE adalah akar kuadrat dari MSE. Ini memberikan perkiraan standar deviasi dari kesalahan prediksi.
   - Rumus RMSE: RMSE = √MSE.
   - Nilai RMSE yang lebih rendah menunjukkan kesalahan prediksi yang lebih kecil dan kualitas prediksi yang lebih baik.

5. R^2 (Coefficient of Determination):
   - R^2 mengukur seberapa baik model regresi cocok dengan data yang diamati.
   - Rumus R^2: R^2 = 1 - (MSE / Var(y_actual)), di mana Var(y_actual) adalah varians dari nilai sebenarnya.
   - Nilai R^2 berkisar antara 0 dan 1. Nilai yang lebih tinggi menunjukkan bahwa model memiliki kemampuan yang lebih baik dalam menjelaskan variasi dalam data.

Dalam konteks hasil kode di atas, hasil evaluasi menunjukkan bahwa model regresi memiliki performa yang baik. MSE, MAE, dan RMSE yang rendah menunjukkan kesalahan prediksi yang kecil, sedangkan MedAE yang rendah menunjukkan kesalahan prediksi yang stabil. Nilai R^2 yang tinggi menunjukkan bahwa model dapat menjelaskan sebagian besar variasi dalam data.
Pada hasil evaluasi modelling data ini menunjukkan MSE, MAE, R-Square (R2), MedAE, dan RMSE pada algoritma SVR adalah: 
- Mean squared error(MSE) =  0.34;
- Mean absolute error(MAE) =  0.43;
- Median absolute error(MedAE) =  0.1;
- RMSE: 0.5848034889023938;
- R^2: 0.912;

Tabel 4. Hasil Matriks Evaluasi 

| Matriks Regresi | Value  |
|-----------------|--------|
| MSE             | 0.34   |
| MAE             | 0.43   |
| MedAE           | 0.1    |
| RMSE            | 0.5848 |
| R^2             | 0.912  |


## KESIMPULAN


Dalam penelitian ini dapat disimpulkan sebagai berikut:

Dari hasil evaluasi yang dilakukan, dapat disimpulkan bahwa model prediksi memiliki kualitas yang baik dan mampu memberikan hasil yang cukup akurat. Berikut adalah analisis lebih lanjut terkait beberapa nilai evaluasi yang diperoleh:

- Mean squared error (MSE) sebesar 0.34 menunjukkan bahwa rata-rata selisih kuadrat antara nilai aktual dan nilai prediksi adalah relatif kecil. Ini menunjukkan tingkat kesalahan yang rendah dalam model prediksi. Mean absolute error (MAE) sebesar 0.43 menunjukkan bahwa rata-rata selisih mutlak antara nilai aktual dan nilai prediksi juga rendah. Artinya, prediksi cenderung mendekati nilai aktual dengan baik.

- Median absolute error (MedAE) sebesar 0.1 mengindikasikan bahwa sebagian besar kesalahan prediksi terletak pada rentang yang relatif kecil. Hal ini menunjukkan konsistensi model dalam memprediksi data dengan tingkat kesalahan yang rendah.

- Root Mean Square Error (RMSE) sebesar 0.5848 menunjukkan akurasi yang tinggi dalam memprediksi nilai aktual. RMSE menggambarkan sejauh mana perbedaan antara nilai aktual dan nilai prediksi secara keseluruhan. Semakin kecil nilai RMSE, semakin akurat prediksi model.

- R-squared (R^2) sebesar 0.912 menandakan bahwa model mampu menjelaskan 91.2% variasi dalam data aktual. Semakin tinggi nilai R^2, semakin baik model dapat menggambarkan pola dan tren yang ada dalam data aktual.

Berdasarkan kesimpulan di atas, dapat dikatakan bahwa model prediksi yang dievaluasi memiliki performa yang baik. Meskipun tidak ada model yang sempurna, nilai-nilai evaluasi yang rendah, seperti MSE, MAE, MedAE, RMSE, dan tingkat keakuratan yang tinggi dengan R^2, menunjukkan bahwa model cenderung memberikan hasil prediksi yang akurat dan konsisten. Namun, tetap perlu dilakukan analisis lebih lanjut untuk memastikan validitas model dan memperhatikan konteks dan tujuan prediksi yang ingin dicapai.


## SARAN

- Evaluasi metrik: Selain MSE, R-squared, dan RMSE, ada baiknya juga melihat metrik evaluasi lainnya seperti MAE (Mean Absolute Error) dan MAPE (Mean Absolute Percentage Error) untuk mendapatkan gambaran yang lebih komprehensif tentang performa model. Metrik evaluasi tambahan ini dapat memberikan wawasan yang lebih lengkap tentang kesalahan model dalam memprediksi nilai target.

- Penyempurnaan model: Mengingat performa yang rendah dalam menjelaskan variasi dalam data, disarankan untuk melakukan penyempurnaan pada model-model yang digunakan. Mungkin perlu mengubah konfigurasi atau parameter pada algoritma KNN, RF, dan NN untuk meningkatkan performa prediksi. Eksperimen dengan berbagai parameter dan teknik tuning dapat membantu meningkatkan hasil prediksi.

- Pemilihan fitur: Perlu dipertimbangkan pemilihan fitur yang lebih relevan dan informatif untuk meningkatkan performa model. Evaluasi ulang fitur yang digunakan dalam model dapat membantu dalam memilih fitur yang lebih penting dan memiliki hubungan yang lebih kuat dengan variabel target.

- Data tambahan: Jika memungkinkan, penambahan data tambahan atau pengumpulan data yang lebih lengkap dan representatif dapat meningkatkan performa model. Dengan memiliki lebih banyak data, model dapat menemukan pola yang lebih baik dan memberikan prediksi yang lebih akurat.

- Model alternatif: Selain algoritma yang telah digunakan, ada baiknya juga mengevaluasi model alternatif. Mungkin ada algoritma lain yang lebih cocok atau memiliki performa yang lebih baik dalam menyelesaikan masalah ini. Mengeksplorasi model lain seperti regresi linear, decision tree, atau ensemble model lainnya dapat memberikan pemahaman yang lebih baik tentang mana model yang paling sesuai untuk dataset ini.

- Validasi ulang: Melakukan validasi ulang terhadap model yang telah ditingkatkan dan melakukan perbandingan dengan model-model alternatif. Validasi silang (cross-validation) atau penggunaan dataset validasi yang lebih besar dapat memberikan kepercayaan yang lebih tinggi terhadap performa model.

Dengan melakukan penyempurnaan pada pemodelan, pemilihan fitur yang tepat, penambahan data, dan evaluasi model alternatif, diharapkan dapat mencapai hasil yang lebih baik dalam memprediksi nilai target dan meningkatkan kemampuan model dalam menjelaskan variasi dalam data.

## DAFTAR ISI

[1] K. M. Hindrayani, I. G. S. Mas Diyasa, P. A. Riyantoko, dan T. M. Fahrudin, “Studi Literatur Mengenai Prediksi Harga Saham Menggunakan Machine Learning,” Pros. Semin. Nas. Inform. Bela Negara, vol. 1, hal. 71–75, 2020, doi: 10.33005/santika.v1i0.20.

[2] R. H. Kusumodestoni dan S. Sarwido, “Komparasi Model Support Vector Machines (Svm) Dan Neural Network Untuk Mengetahui Tingkat Akurasi Prediksi Tertinggi Harga Saham,” J. Inform. Upgris, vol. 3, no. 1, 2017, doi: 10.26877/jiu.v3i1.1536.

[3] A. B. Untoro, “Prediksi Harga Saham Dengan Menggunakan Jaringan Syaraf Tiruan”, Jurnal Teknologi Informatika dan Komputer MH Thamrin, vol. 6, no. 2, pp.103-111, 2020.

[4] Patriya, “Implementasi Support Vector Machine Pada Prediksi Harga Saham Gabungan (IHSG),” Jurnal Ilmiah Teknologi dan Rekayasa, vol. 25, no. 1, pp. 24–38, 2020.

[5] F. R. Setiawan, R. F. Umbara, and A.A. Rohmawati, “Prediksi Pergerakan Harga Saham dengan Metode Support Vector Machine (SVM) Menggunakan Trend Deterministic Data Preparation”, e-Proxeeding of Engineering, vol. 5m no. 3, pp.8356-8372, 2018.

[6] R. Akmalia, I. Slamet, dan H. Pratiwi, “Analisis Sentimen Twitter Berbahasa Indonesia Terhadap Aplikasi,” Pros. Semin. Nas. MIPA UNIPA, vol. 2022, hal. 150–156, 2022.

[7] R. Primartha, Algoritma Machine Learning. Informatika, 2021.

[8] R. T. Handayanto dan Herlawati, Data Mining dan Machine Learning Menggunakan Matlab & Python. Penerbit informatika, 2020.



```python

```
