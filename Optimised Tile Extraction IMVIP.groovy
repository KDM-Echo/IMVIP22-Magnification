import qupath.lib.objects.PathObjects
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import groovy.time.TimeCategory 
import groovy.time.TimeDuration
Annos = getAnnotationObjects()

if (Annos.size() >=0){

    
    Date start = new Date()
    // Set Desired Output
    ImageOverlapPercentage = 0 // Between 0 and 1
    OutputImageSize = 512
    OutputImageMagnification = 2
    
    
    UniqueName = "512px - 2x Mag - 0% Overlap with binary - Kris - Immature"
    
    // Get Image Information
    
    imageData = getCurrentImageData()
    server = imageData.getServer()
    
    
    ImageName = server.getMetadata().getName()
    ImageWidth = server.getWidth()
    ImageHeight = server.getHeight()
    PixelSize = server.getPixelCalibration().getAveragedPixelSize()
    ImageMagnification = server.getOriginalMetadata().getMagnification()
    

    
    //print(ImageName)
    //print(ImageWidth)
    //print(ImageHeight)
    //print(PixelSize)
    //print(ImageMagnification)
    
    // Calculate the downsample value
    
    Downsample = ImageMagnification / OutputImageMagnification
    //print(Downsample)
    
    // Get the pixel size required for each tile on the image.
    
    TileSizePixels = (OutputImageSize * Downsample).toInteger()
    //print(TileSizePixels)
    
    patch_um = TileSizePixels * PixelSize
    
    
    // Create Tiles accross the whole image
    
    Start_Y = 0
    
    XCoList = []
    YCoList = []
    
    // Get lists of valid top left coordinates for tiles
    while (Start_Y < ImageHeight){
    
        Start_X = 0
        
        while (Start_X < ImageWidth){
        
            End_X = Start_X + TileSizePixels
            End_Y = Start_Y + TileSizePixels
            
            
            if (End_X < ImageWidth && End_Y < ImageHeight){
                XCoList.add(Start_X)
                YCoList.add(Start_Y)}
            
            Start_X += (TileSizePixels*(1-ImageOverlapPercentage)).toInteger()}
            
        Start_Y += (TileSizePixels*(1-ImageOverlapPercentage)).toInteger()}
        
          
           
    TLSAnnos = getObjects({p -> p.isAnnotation() == true && p.getPathClass() != null  && p.getPathClass() != getPathClass("TLS with GC")})
    
    
    
    // Create the folders for each class
    OriginalsFolder = "E:/" + UniqueName + "/Original Images/" + ImageName + "/"
    TLSFolder = "E:/TLS - " + UniqueName + "/Original Images/" + ImageName + "/" 
    BinFolder = "E:/" + UniqueName + "/Binary_Masks/" + ImageName + "/"
    TLSFolderBin = "E:/TLS - " + UniqueName + "/Binary_Masks/" + ImageName + "/"

    mkdirs(OriginalsFolder)
    mkdirs(BinFolder)
    mkdirs(TLSFolder)
    mkdirs(TLSFolderBin)

    
    
    def labelServer = new LabeledImageServer.Builder(imageData)
        .backgroundLabel(255, ColorTools.BLACK)
        .addLabel("Stroma",1)        //Set to class you want as white on binary mask
        .downsample(Downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
        .multichannelOutput(false)  // If true, each label is a different channel (required for multiclass probability)
        .build()
    
    
    for(int x=0; x<=XCoList.size()-1; x++){
        xLOC = XCoList[x]
        yLOC = YCoList[x]
        
        
        //RectA
        AX1 = xLOC
        AY1 = yLOC
        AX2 = xLOC + TileSizePixels
        AY2 = yLOC + TileSizePixels
    
        
        
    
        
        
        
        
        if (TLSAnnos.size() >0){    
            for (t in TLSAnnos){
                //RectB
                BX1 = (t.getROI().getCentroidX()) - (TileSizePixels/2)
                BY1 = (t.getROI().getCentroidY()) - (TileSizePixels/2)
                BX2 = (t.getROI().getCentroidX()) + (TileSizePixels/2)
                BY2 = (t.getROI().getCentroidY()) + (TileSizePixels/2)
                
                
        
               
               
            
                //if (BottomX > xLOC && xLOC > TopX && BottomY > yLOC && yLOC > TopY){
                //    t.setPathClass(T.getPathClass())}
                
                if  (BX2 < AX1 || BY1 > AY2 || BX1 > AX2 || BY2 < AY1){ 
                    TLS = false
    
                // No overlap, no TLS in ROI
                    
                 
                    
                    }
                    
                else{
                // Overlap, TLS in Tile
                    TLS = true
    
                    break
                    }
                }
    
            if(TLS){
    
                fullTileName =  ImageName + "_NoTLS" + "_(" + Downsample + "," + xLOC + "," + yLOC + "," + TileSizePixels + "," + TileSizePixels + ").png"
                fullTileNameBin =  ImageName + "_NoTLS" + "_(" + Downsample + "," + xLOC + "," + yLOC + "," + TileSizePixels + "," + TileSizePixels + ")-mask.png"  
                GroupedName = OriginalsFolder + ImageName + "_NoTLS" + "_(" + Downsample + "," + xLOC + "," + yLOC + "," + TileSizePixels + "," + TileSizePixels + ").png"
                GroupedNameBin = BinFolder + ImageName + "_NoTLS" + "_(" + Downsample + "," + xLOC + "," + yLOC + "," + TileSizePixels + "," + TileSizePixels + ")-mask.png"
                SaveLoc = TLSFolder + fullTileName
                SaveLocBin = TLSFolderBin + fullTileNameBin}
            else{
    
                fullTileName = ImageName + "_NoTLS" + "_(" + Downsample + "," + xLOC + "," + yLOC + "," + TileSizePixels + "," + TileSizePixels + ").png"
                fullTileNameBin =  ImageName + "_NoTLS" + "_(" + Downsample + "," + xLOC + "," + yLOC + "," + TileSizePixels + "," + TileSizePixels + ")-mask.png"  
                GroupedName = OriginalsFolder + ImageName + "_NoTLS" + "_(" + Downsample + "," + xLOC + "," + yLOC + "," + TileSizePixels + "," + TileSizePixels + ").png"
                GroupedNameBin = BinFolder + ImageName + "_NoTLS" + "_(" + Downsample + "," + xLOC + "," + yLOC + "," + TileSizePixels + "," + TileSizePixels + ")-mask.png"  
                //SaveLoc = NegativeFolder + fullTileName
               //SaveLocBin = NegativeFolderBin + fullTileNameBin
               }
                
                    
                    
                    
                request = RegionRequest.createInstance(imageData.getServerPath(), Downsample, xLOC, yLOC, TileSizePixels, TileSizePixels)
               /// ImageWriterTools.writeImageRegion(server, request, SaveLoc)
               /// ImageWriterTools.writeImageRegion(server, request, GroupedName)
            
                // save Binary mask
                def region = RegionRequest.createInstance(labelServer.getPath(), Downsample, xLOC, yLOC, TileSizePixels, TileSizePixels)
                //ImageWriterTools.writeImageRegion(labelServer, region, SaveLocBin)
               // ImageWriterTools.writeImageRegion(labelServer, region, GroupedNameBin)
            }
        
        else{
            fullTileName = ImageName + "_NoTLS" + "_(" + Downsample + "," + xLOC + "," + yLOC + "," + TileSizePixels + "," + TileSizePixels + ").png"
            fullTileNameBin =  ImageName + "_NoTLS" + "_(" + Downsample + "," + xLOC + "," + yLOC + "," + TileSizePixels + "," + TileSizePixels + ")-mask.png"  
            GroupedName = OriginalsFolder + ImageName + "_NoTLS" + "_(" + Downsample + "," + xLOC + "," + yLOC + "," + TileSizePixels + "," + TileSizePixels + ").png"
            GroupedNameBin = BinFolder + ImageName + "_NoTLS" + "_(" + Downsample + "," + xLOC + "," + yLOC + "," + TileSizePixels + "," + TileSizePixels + ")-mask.png"  
            //SaveLoc = NegativeFolder + fullTileName
            }
            //SaveLocBin = NegativeFolderBin + fullTileNameBin}
            
        request = RegionRequest.createInstance(imageData.getServerPath(), Downsample, xLOC, yLOC, TileSizePixels, TileSizePixels)
        //ImageWriterTools.writeImageRegion(server, request, SaveLoc)
        ImageWriterTools.writeImageRegion(server, request, GroupedName)
            
        // save Binary mask
        def region = RegionRequest.createInstance(labelServer.getPath(), Downsample, xLOC, yLOC, TileSizePixels, TileSizePixels)
        //ImageWriterTools.writeImageRegion(labelServer, region, SaveLocBin)
        ImageWriterTools.writeImageRegion(labelServer, region, GroupedNameBin)}
        
            
        
                
            
    print("Done")
    Date stop = new Date()
    
    TimeDuration td = TimeCategory.minus( stop, start )
    print(td)}
    
else{
;}
    
        
    

    