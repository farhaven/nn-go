package main

import (
	"encoding/binary"
	"io"
	"log"
	"os"
)

func ReadMnist(prefix string) []Sample {
	log.Println(`reading mnist data from`, prefix)

	imgfh, err := os.Open(`mnist/` + prefix + `-images-idx3-ubyte`)
	if err != nil {
		log.Fatalln(`can't open images:`, err)
	}
	defer imgfh.Close()

	lblfh, err := os.Open(`mnist/` + prefix + `-labels-idx1-ubyte`)
	if err != nil {
		log.Fatalln(`can't open images:`, err)
	}
	defer lblfh.Close()

	/* Read headers */
	var imgMagic uint32
	err = binary.Read(imgfh, binary.BigEndian, &imgMagic)
	if err != nil {
		log.Fatalln(`can't read image header:`, err)
	}
	if imgMagic != 0x0803 {
		log.Fatalln(`wrong magic`)
	}
	log.Println(`got img magic`, imgMagic)

	var lblMagic uint32
	err = binary.Read(lblfh, binary.BigEndian, &lblMagic)
	if err != nil {
		log.Fatalln(`can't read image header:`, err)
	}
	if lblMagic != 0x0801 {
		log.Fatalln(`wrong magic`)
	}
	log.Println(`got lbl magic`, lblMagic)

	var numImgs uint32
	err = binary.Read(imgfh, binary.BigEndian, &numImgs)
	if err != nil {
		log.Fatalln(`can't read image header:`, err)
	}
	log.Println(`got number of imgs`, numImgs)

	var numLbls uint32
	err = binary.Read(lblfh, binary.BigEndian, &numLbls)
	if err != nil {
		log.Fatalln(`can't read image header:`, err)
	}
	log.Println(`got number of imgs`, numLbls)

	if numImgs != numLbls {
		log.Fatalln(`mismatch between number of images and labels`)
	}

	var imgDims [2]uint32
	err = binary.Read(imgfh, binary.BigEndian, &imgDims)
	if err != nil {
		log.Fatalln(`can't read image header:`, err)
	}
	log.Println(`image dimensions`, imgDims)

	if imgDims != [2]uint32{28, 28} {
		log.Fatalln(`unexpected image dimensions`)
	}

	/* Load images and labels and build samples from that */
	buf := make([]uint8, imgDims[0]*imgDims[1])
	samples := []Sample{}
	for {
		err = binary.Read(imgfh, binary.BigEndian, buf)
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatalln(`can't read:`, err)
		}
		img := make([]float64, len(buf))
		for idx, val := range buf {
			img[idx] = float64(val) / 255.0
		}

		onehot := make([]float64, 10)
		var label uint8
		err = binary.Read(lblfh, binary.BigEndian, &label)
		if label >= 10 {
			log.Fatalln(`unexpected label:`, label)
		}
		onehot[label] = 1.0

		samples = append(samples, Sample{
			inputs: img,
			targets: onehot,
		})
	}

	return samples
}
