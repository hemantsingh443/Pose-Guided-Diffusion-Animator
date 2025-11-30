from PIL import Image, ImageDraw

class StickFigureRenderer:
    def __init__(self, image_size=(128, 128), line_width=2):
        self.image_size = image_size
        self.line_width = line_width
        
    def render(self, joints, skeleton_config=None):
        img = Image.new('RGB', self.image_size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        W, H = self.image_size
        
        def to_pix(pt):
            return (pt[0] * W, pt[1] * H)
            
        draw.line([to_pix(joints['hip']), to_pix(joints['neck'])], fill=(0,0,0), width=self.line_width)
        
        r_norm = 0.05
        if skeleton_config and 'head_radius' in skeleton_config.lengths:
            r_norm = skeleton_config.lengths['head_radius']
            
        r = r_norm * H 
        hx, hy = to_pix(joints['head'])
        draw.ellipse([hx-r, hy-r, hx+r, hy+r], outline=(0,0,0), width=self.line_width)
        
        draw.line([to_pix(joints['neck']), to_pix(joints['l_elbow'])], fill=(0,0,0), width=self.line_width)
        draw.line([to_pix(joints['l_elbow']), to_pix(joints['l_hand'])], fill=(0,0,0), width=self.line_width)
        
        draw.line([to_pix(joints['neck']), to_pix(joints['r_elbow'])], fill=(0,0,0), width=self.line_width)
        draw.line([to_pix(joints['r_elbow']), to_pix(joints['r_hand'])], fill=(0,0,0), width=self.line_width)
        
        draw.line([to_pix(joints['hip']), to_pix(joints['l_knee'])], fill=(0,0,0), width=self.line_width)
        draw.line([to_pix(joints['l_knee']), to_pix(joints['l_foot'])], fill=(0,0,0), width=self.line_width)
        
        draw.line([to_pix(joints['hip']), to_pix(joints['r_knee'])], fill=(0,0,0), width=self.line_width)
        draw.line([to_pix(joints['r_knee']), to_pix(joints['r_foot'])], fill=(0,0,0), width=self.line_width)
        
        return img
