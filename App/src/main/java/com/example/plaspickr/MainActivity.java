package com.example.plaspickr;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.plaspickr.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {

    private ImageView imgView, myImageView;
    private Button select, predict, predwaste;
    private TextView tv;
    private Bitmap img;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imgView = (ImageView) findViewById(R.id.imageView);
        myImageView = (ImageView) findViewById(R.id.imageView2);
        tv = (TextView) findViewById(R.id.textView);
        select = (Button) findViewById(R.id.button);
        predict = (Button) findViewById(R.id.button3);
        predwaste = (Button) findViewById(R.id.button2);


        select.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 100);

            }
        });

        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                img = Bitmap.createScaledBitmap(img, 180, 180, true);

                try {
                    Model model = Model.newInstance(getApplicationContext());
                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 180, 180, 3}, DataType.FLOAT32);

                    TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
                    tensorImage.load(img);
                    ByteBuffer byteBuffer = tensorImage.getBuffer();

                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    Model.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
                    float[] confidences = outputFeature0.getFloatArray();
                    int maxPos = 0;
                    float maxConfidence = 0;
                    for (int i = 0; i < confidences.length; i++) {
                        if (confidences[i] > maxConfidence) {
                            maxConfidence = confidences[i];
                            maxPos = i;
                        }
                    }
                    String[] classes = {"HDPE", "LDPE", "Other", "PET", "PP", "PS", "PVC"};
                    if(classes[maxPos]=="PET"){
                        myImageView.setImageResource(R.drawable.pet);
                    }
                    else if(classes[maxPos]=="HDPE"){
                        myImageView.setImageResource(R.drawable.hdpe);
                    }
                    else if(classes[maxPos]=="PVC"){
                        myImageView.setImageResource(R.drawable.pvc);
                    }
                    else if(classes[maxPos]=="LDPE"){
                        myImageView.setImageResource(R.drawable.ldpe);
                    }
                    else if(classes[maxPos]=="PP"){
                        myImageView.setImageResource(R.drawable.pp);
                    }
                    else if(classes[maxPos]=="PS"){
                        myImageView.setImageResource(R.drawable.ps);
                    }
                    else {
                        myImageView.setImageResource(R.drawable.other);
                    }
                    // Releases model resources if no longer used.
                    model.close();


//                    tv.setText(outputFeature0.getFloatArray()[0]
//                            + "\n"+outputFeature0.getFloatArray()[1]+
//                            "\n"+outputFeature0.getFloatArray()[2]+
//                            "\n"+outputFeature0.getFloatArray()[3]+
//                            "\n"+outputFeature0.getFloatArray()[4]+
//                            "\n"+outputFeature0.getFloatArray()[5]+
//                            "\n"+outputFeature0.getFloatArray()[6]);
                    tv.setText(classes[maxPos]);


                } catch (IOException e) {
                    // TODO Handle the exception
                }

            }
        });

        predwaste.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                img = Bitmap.createScaledBitmap(img, 180, 180, true);

                try {
                    Model model1 = Model.newInstance(getApplicationContext());
                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 180, 180, 3}, DataType.FLOAT32);

                    TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
                    tensorImage.load(img);
                    ByteBuffer byteBuffer = tensorImage.getBuffer();

                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    Model.Outputs outputs = model1.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
                    float[] confidences = outputFeature0.getFloatArray();
                    int maxPos = 0;
                    float maxConfidence = 0;
                    for (int i = 0; i < confidences.length; i++) {
                        if (confidences[i] > maxConfidence) {
                            maxConfidence = confidences[i];
                            maxPos = i;
                        }
                    }
                    String[] classes = {"cardboard", "e-waste", "glass", "metal", "paper", "plastic", "trash"};
                    if(classes[maxPos]=="cardboard"){
                        myImageView.setImageResource(R.drawable.cardboard);
                    }
                    else if(classes[maxPos]=="e-waste"){
                        myImageView.setImageResource(R.drawable.ewaste);
                    }
                    else if(classes[maxPos]=="glass"){
                        myImageView.setImageResource(R.drawable.glass);
                    }
                    else if(classes[maxPos]=="metal"){
                        myImageView.setImageResource(R.drawable.metal);
                    }
                    else if(classes[maxPos]=="paper"){
                        myImageView.setImageResource(R.drawable.paper);
                    }
                    else if(classes[maxPos]=="plastic"){
                        myImageView.setImageResource(R.drawable.plastic);
                    }
                    else {
                        myImageView.setImageResource(R.drawable.trash);
                    }
                    // Releases model resources if no longer used.
                    model1.close();


    //                    tv.setText(outputFeature0.getFloatArray()[0]
    //                            + "\n"+outputFeature0.getFloatArray()[1]+
    //                            "\n"+outputFeature0.getFloatArray()[2]+
    //                            "\n"+outputFeature0.getFloatArray()[3]+
    //                            "\n"+outputFeature0.getFloatArray()[4]+
    //                            "\n"+outputFeature0.getFloatArray()[5]+
    //                            "\n"+outputFeature0.getFloatArray()[6]);
                    tv.setText(classes[maxPos]);


                } catch (IOException e) {
                    // TODO Handle the exception
                }

            }
        });

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode == 100)
        {
            imgView.setImageURI(data.getData());

            Uri uri = data.getData();
            try {
                img = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
