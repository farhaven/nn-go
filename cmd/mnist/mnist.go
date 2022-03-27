package main

import (
	"encoding/binary"
	"io"
	"log"
	"os"
)

type mnistSample struct {
	input  []float64
	target []float64
}

func readMnist(prefix string) []mnistSample {
	logger := log.New(os.Stdout, `[MNIST] `, log.LstdFlags)

	logger.Println(`reading mnist data from`, prefix)

	imgfh, err := os.Open(`mnist/` + prefix + `-images-idx3-ubyte`)
	if err != nil {
		logger.Fatalln(`can't open images:`, err)
	}
	defer imgfh.Close()

	lblfh, err := os.Open(`mnist/` + prefix + `-labels-idx1-ubyte`)
	if err != nil {
		logger.Fatalln(`can't open images:`, err)
	}
	defer lblfh.Close()

	/* Read headers */
	var imgMagic uint32
	err = binary.Read(imgfh, binary.BigEndian, &imgMagic)
	if err != nil {
		logger.Fatalln(`can't read image header:`, err)
	}
	if imgMagic != 0x0803 {
		logger.Fatalln(`wrong magic`)
	}
	logger.Println(`got img magic`, imgMagic)

	var lblMagic uint32
	err = binary.Read(lblfh, binary.BigEndian, &lblMagic)
	if err != nil {
		logger.Fatalln(`can't read image header:`, err)
	}
	if lblMagic != 0x0801 {
		logger.Fatalln(`wrong magic`)
	}
	logger.Println(`got lbl magic`, lblMagic)

	var numImgs uint32
	err = binary.Read(imgfh, binary.BigEndian, &numImgs)
	if err != nil {
		logger.Fatalln(`can't read image header:`, err)
	}
	logger.Println(`got number of imgs`, numImgs)

	var numLbls uint32
	err = binary.Read(lblfh, binary.BigEndian, &numLbls)
	if err != nil {
		logger.Fatalln(`can't read image header:`, err)
	}
	logger.Println(`got number of imgs`, numLbls)

	if numImgs != numLbls {
		logger.Fatalln(`mismatch between number of images and labels`)
	}

	var imgDims [2]uint32
	err = binary.Read(imgfh, binary.BigEndian, &imgDims)
	if err != nil {
		logger.Fatalln(`can't read image header:`, err)
	}
	logger.Println(`image dimensions`, imgDims)

	if imgDims != [2]uint32{28, 28} {
		logger.Fatalln(`unexpected image dimensions`)
	}

	/* Load images and labels and build samples from that */
	buf := make([]uint8, imgDims[0]*imgDims[1])
	samples := []mnistSample{}
	for {
		err = binary.Read(imgfh, binary.BigEndian, buf)
		if err == io.EOF {
			break
		} else if err != nil {
			logger.Fatalln(`can't read:`, err)
		}
		img := make([]float64, len(buf))
		for idx, val := range buf {
			img[idx] = float64(val) / 255.0
		}

		onehot := make([]float64, 10)
		var label uint8
		err = binary.Read(lblfh, binary.BigEndian, &label)
		if err != nil {
			logger.Fatalln("can't read label")
		}

		if label >= 10 {
			logger.Fatalln(`unexpected label:`, label)
		}
		onehot[label] = 1.0

		samples = append(samples, mnistSample{
			input:  img,
			target: onehot,
		})
	}

	return samples
}
