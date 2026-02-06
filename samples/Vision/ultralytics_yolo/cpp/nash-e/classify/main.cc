/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

Copyright (c) 2024-2025, D-Robotics.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// 注意: 此程序在RDK S100 (Nash-E) 板端运行
// Attention: This program runs on RDK S100 (Nash-E) board.

// ============================================================================ 
// Configuration Parameters
// ============================================================================ 

#define MODEL_PATH "yolo11s_cls_nashe_640x640_nv12.hbm"
#define TEST_IMG_PATH "../../../../../../resource/assets/zebra_cls.jpg"

// 前处理方式: 0=Resize, 1=LetterBox
#define RESIZE_TYPE 0
#define LETTERBOX_TYPE 1
#define PREPROCESS_TYPE LETTERBOX_TYPE

#define TOP_K 5

// ============================================================================ 
// Includes
// ============================================================================ 

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <opencv2/opencv.hpp>
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

// ============================================================================ 
// Macros & Helpers
// ============================================================================ 

#define CHECK_SUCCESS(value, errmsg)                                         \
    do {                                                                     \
        auto ret_code = value;                                               \
        if (ret_code != 0) {                                                 \
            std::cerr << "\033[1;31m[ERROR]\033[0m " << __FILE__ << ":"     \
                      << __LINE__ << " " << errmsg                           \
                      << ", error code: " << ret_code << std::endl;          \
            return ret_code;                                                 \
        }                                                                    \
    } while (0)

#define LOG_INFO(msg) \
    std::cout << "\033[1;32m[INFO]\033[0m " << msg << std::endl

#define LOG_WARN(msg) \
    std::cout << "\033[1;33m[WARN]\033[0m " << msg << std::endl

#define LOG_ERROR(msg) \
    std::cerr << "\033[1;31m[ERROR]\033[0m " << msg << std::endl

#define LOG_TIME(msg, duration) \
    std::cout << "\033[1;31m" << msg << " = " << std::fixed            \
              << std::setprecision(2) << (duration) << " ms\033[0m"    \
              << std::endl

const std::vector<std::string> IMAGENET_CLASSES = {
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark",
    "electric ray", "stingray", "cock", "hen", "ostrich",
    "brambling", "goldfinch", "house finch", "junco", "indigo bunting",
    "robin", "bulbul", "jay", "magpie", "chickadee",
    "water ouzel", "kite", "bald eagle", "vulture", "great grey owl",
    "European fire salamander", "common newt", "eft", "spotted salamander", "axolotl",
    "bullfrog", "tree frog", "tailed frog", "loggerhead", "leatherback turtle",
    "mud turtle", "terrapin", "box turtle", "banded gecko", "common iguana",
    "American chameleon", "whiptail", "agama", "frilled lizard", "alligator lizard",
    "Gila monster", "green lizard", "African chameleon", "Komodo dragon", "African crocodile",
    "American alligator", "triceratops", "thunder snake", "ringneck snake", "hognose snake",
    "green snake", "king snake", "garter snake", "water snake", "vine snake",
    "night snake", "boa constrictor", "rock python", "Indian cobra", "green mamba",
    "sea snake", "horned viper", "diamondback", "sidewinder", "trilobite",
    "harvestman", "scorpion", "black and gold garden spider", "barn spider", "garden spider",
    "black widow", "tarantula", "wolf spider", "tick", "centipede",
    "black grouse", "ptarmigan", "ruffed grouse", "prairie chicken", "peacock",
    "quail", "partridge", "African grey", "macaw", "sulphur-crested cockatoo",
    "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird",
    "jacamar", "toucan", "drake", "red-breasted merganser", "goose",
    "black swan", "tusker", "echidna", "platypus", "wallaby",
    "koala", "wombat", "jellyfish", "sea anemone", "brain coral",
    "flatworm", "nematode", "conch", "snail", "slug",
    "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
    "fiddler crab", "king crab", "American lobster", "spiny lobster", "crayfish",
    "hermit crab", "isopod", "white stork", "black stork", "spoonbill",
    "flamingo", "little blue heron", "American egret", "bittern", "crane",
    "limpkin", "European gallinule", "American coot", "bustard", "ruddy turnstone",
    "red-backed sandpiper", "redshank", "dowitcher", "oystercatcher", "pelican",
    "king penguin", "albatross", "grey whale", "killer whale", "dugong",
    "sea lion", "Chihuahua", "Japanese spaniel", "Maltese dog", "Pekinese",
    "Shih-Tzu", "Blenheim spaniel", "papillon", "toy terrier", "Rhodesian ridgeback",
    "Afghan hound", "basset", "beagle", "bloodhound", "bluetick",
    "black-and-tan coonhound", "Walker hound", "English foxhound", "redbone", "borzoi",
    "Irish wolfhound", "Italian greyhound", "whippet", "Ibizan hound", "Norwegian elkhound",
    "otterhound", "Saluki", "Scottish deerhound", "Weimaraner", "Staffordshire bullterrier",
    "American Staffordshire terrier", "Bedlington terrier", "Border terrier", "Kerry blue terrier", "Irish terrier",
    "Norfolk terrier", "Norwich terrier", "Yorkshire terrier", "wire-haired fox terrier", "Lakeland terrier",
    "Sealyham terrier", "Airedale", "cairn", "Australian terrier", "Dandie Dinmont",
    "Boston bull", "miniature schnauzer", "giant schnauzer", "standard schnauzer", "Scotch terrier",
    "Tibetan terrier", "silky terrier", "soft-coated wheaten terrier", "West Highland white terrier", "Lhasa",
    "flat-coated retriever", "curly-coated retriever", "golden retriever", "Labrador retriever", "Chesapeake Bay retriever",
    "German short-haired pointer", "vizsla", "English setter", "Irish setter", "Gordon setter",
    "Brittany spaniel", "clumber", "English springer", "Welsh springer spaniel", "cocker spaniel",
    "Sussex spaniel", "Irish water spaniel", "kuvasz", "schipperke", "groenendael",
    "malinois", "briard", "kelpie", "komondor", "Old English sheepdog",
    "Shetland sheepdog", "collie", "Border collie", "Bouvier des Flandres", "Rottweiler",
    "German shepherd", "Doberman", "miniature pinscher", "Greater Swiss Mountain dog", "Bernese mountain dog",
    "Appenzeller", "EntleBucher", "boxer", "bull mastiff", "Tibetan mastiff",
    "French bulldog", "Great Dane", "Saint Bernard", "Eskimo dog", "malamute",
    "Siberian husky", "dalmatian", "affenpinscher", "basenji", "pug",
    "Leonberg", "Newfoundland", "Great Pyrenees", "Samoyed", "Pomeranian",
    "chow", "keeshond", "Brabancon griffon", "Pembroke", "Cardigan",
    "toy poodle", "miniature poodle", "standard poodle", "Mexican hairless", "timber wolf",
    "white wolf", "red wolf", "coyote", "dingo", "dhole",
    "African hunting dog", "hyena", "red fox", "kit fox", "Arctic fox",
    "grey fox", "tabby", "tiger cat", "Persian cat", "Siamese cat",
    "Egyptian cat", "cougar", "lynx", "leopard", "snow leopard",
    "jaguar", "lion", "tiger", "cheetah", "brown bear",
    "American black bear", "ice bear", "sloth bear", "mongoose", "meerkat",
    "tiger beetle", "ladybug", "ground beetle", "long-horned beetle", "leaf beetle",
    "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee",
    "ant", "grasshopper", "cricket", "walking stick", "cockroach",
    "mantis", "cicada", "leafhopper", "lacewing", "dragonfly",
    "damselfly", "admiral", "ringlet", "monarch", "cabbage butterfly",
    "sulphur butterfly", "lycaenid", "starfish", "sea urchin", "sea cucumber",
    "wood rabbit", "hare", "Angora", "hamster", "porcupine",
    "fox squirrel", "marmot", "beaver", "guinea pig", "sorrel",
    "zebra", "hog", "wild boar", "warthog", "hippopotamus",
    "ox", "water buffalo", "bison", "ram", "bighorn",
    "ibex", "hartebeest", "impala", "gazelle", "Arabian camel",
    "llama", "weasel", "mink", "polecat", "black-footed ferret",
    "otter", "skunk", "badger", "armadillo", "three-toed sloth",
    "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang",
    "guenon", "patas", "baboon", "macaque", "langur",
    "colobus", "proboscis monkey", "marmoset", "capuchin", "howler monkey",
    "titi", "spider monkey", "squirrel monkey", "Madagascar cat", "indri",
    "Indian elephant", "African elephant", "lesser panda", "giant panda", "barracouta",
    "eel", "coho", "rock beauty", "anemone fish", "sturgeon",
    "gar", "lionfish", "puffer", "abacus", "abaya",
    "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner",
    "airship", "altar", "ambulance", "amphibian", "analog clock",
    "apiary", "apron", "ashcan", "assault rifle", "backpack",
    "bakery", "balance beam", "balloon", "ballpoint", "Band Aid",
    "banjo", "bannister", "barbell", "barber chair", "barbershop",
    "barn", "barometer", "barrel", "barrow", "baseball",
    "basketball", "bassinet", "bassoon", "bathing cap", "bath towel",
    "bathtub", "beach wagon", "beacon", "beaker", "bearskin",
    "beer bottle", "beer glass", "bell cote", "bib", "bicycle-built-for-two",
    "bikini", "binder", "binoculars", "birdhouse", "boathouse",
    "bobsled", "bolo tie", "bonnet", "bookcase", "bookshop",
    "bottlecap", "bow", "bow tie", "brass", "brassiere",
    "breakwater", "breastplate", "broom", "bucket", "buckle",
    "bulletproof vest", "bullet train", "butcher shop", "cab", "caldron",
    "candle", "cannon", "canoe", "can opener", "cardigan",
    "car mirror", "carousel", "carpenter's kit", "carton", "car wheel",
    "cash machine", "cassette", "cassette player", "castle", "catamaran",
    "CD player", "cello", "cellular telephone", "chain", "chainlink fence",
    "chain mail", "chain saw", "chest", "chiffonier", "chime",
    "china cabinet", "Christmas stocking", "church", "cinema", "cleaver",
    "cliff dwelling", "cloak", "clog", "cocktail shaker", "coffee mug",
    "coffeepot", "coil", "combination lock", "computer keyboard", "confectionery",
    "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
    "cowboy hat", "cradle", "crane2", "crash helmet", "crate",
    "crib", "Crock Pot", "croquet ball", "crutch", "cuirass",
    "dam", "desk", "desktop computer", "dial telephone", "diaper",
    "digital clock", "digital watch", "dining table", "dishrag", "dishwasher",
    "disk brake", "dock", "dogsled", "dome", "doormat",
    "drilling platform", "drum", "drumstick", "dumbbell", "Dutch oven",
    "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope",
    "espresso maker", "face powder", "feather boa", "file", "fireboat",
    "fire engine", "fire screen", "flagpole", "flute", "folding chair",
    "football helmet", "forklift", "fountain", "fountain pen", "four-poster",
    "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
    "gasmask", "gas pump", "goblet", "go-kart", "golf ball",
    "golfcart", "gondola", "gong", "gown", "grand piano",
    "greenhouse", "grille", "grocery store", "guillotine", "hair slide",
    "hair spray", "half track", "hammer", "hamper", "hand blower",
    "hand-held computer", "handkerchief", "hard disc", "harmonica", "harp",
    "harvester", "hatchet", "holster", "home theater", "honeycomb",
    "hook", "hoopskirt", "horizontal bar", "horse cart", "hourglass",
    "iPod", "iron", "jack-o'-lantern", "jean", "jeep",
    "jersey", "jigsaw puzzle", "jinrikisha", "joystick", "kimono",
    "knee pad", "knot", "lab coat", "ladle", "lampshade",
    "laptop", "lawn mower", "lens cap", "letter opener", "library",
    "lifeboat", "lighter", "limousine", "liner", "lipstick",
    "Loafer", "lotion", "loudspeaker", "loupe", "lumbermill",
    "magnetic compass", "mailbag", "mailbox", "maillot", "maillot",
    "manhole cover", "maraca", "marimba", "mask", "matchstick",
    "maypole", "maze", "measuring cup", "medicine chest", "megalith",
    "microphone", "microwave", "military uniform", "milk can", "minibus",
    "miniskirt", "minivan", "missile", "mitten", "mixing bowl",
    "mobile home", "Model T", "modem", "monastery", "monitor",
    "moped", "mortar", "mortarboard", "mosque", "mosquito net",
    "motor scooter", "mountain bike", "mountain tent", "mouse", "mousetrap",
    "moving van", "muzzle", "nail", "neck brace", "necklace",
    "nipple", "notebook", "obelisk", "oboe", "ocarina",
    "odometer", "oil filter", "organ", "oscilloscope", "overskirt",
    "oxcart", "oxygen mask", "packet", "paddle", "paddlewheel",
    "padlock", "paintbrush", "pajama", "palace", "panpipe",
    "paper towel", "parachute", "parallel bars", "park bench", "parking meter",
    "passenger car", "patio", "pay-phone", "pedestal", "pencil box",
    "pencil sharpener", "perfume", "Petri dish", "photocopier", "pick",
    "pickelhaube", "picket fence", "pickup", "pier", "piggy bank",
    "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate",
    "pitcher", "plane", "planetarium", "plastic bag", "plate rack",
    "plow", "plunger", "Polaroid camera", "pole", "police van",
    "poncho", "pool table", "pop bottle", "pot", "potter's wheel",
    "power drill", "prayer rug", "printer", "prison", "projectile",
    "projector", "puck", "punching bag", "purse", "quill",
    "quilt", "racer", "racket", "radiator", "radio",
    "radio telescope", "rain barrel", "recreational vehicle", "reel", "reflex camera",
    "refrigerator", "remote control", "restaurant", "revolver", "rifle",
    "rocking chair", "rotisserie", "rubber eraser", "rugby ball", "rule",
    "running shoe", "safe", "safety pin", "saltshaker", "sandal",
    "sarong", "sax", "scabbard", "scale", "school bus",
    "schooner", "scoreboard", "screen", "screw", "screwdriver",
    "seat belt", "sewing machine", "shield", "shoe shop", "shoji",
    "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain",
    "ski", "ski mask", "sleeping bag", "slide rule", "sliding door",
    "slot", "snorkel", "snowmobile", "snowplow", "soap dispenser",
    "soccer ball", "sock", "solar dish", "sombrero", "soup bowl",
    "space bar", "space heater", "space shuttle", "spatula", "speedboat",
    "spider web", "spindle", "sports car", "spotlight", "stage",
    "steam locomotive", "steel arch bridge", "steel drum", "stethoscope", "stole",
    "stone wall", "stopwatch", "stove", "strainer", "streetcar",
    "stretcher", "studio couch", "stupa", "submarine", "suit",
    "sundial", "sunglass", "sunglasses", "sunscreen", "suspension bridge",
    "swab", "sweatshirt", "swimming trunks", "swing", "switch",
    "syringe", "table lamp", "tank", "tape player", "teapot",
    "teddy", "television", "tennis ball", "thatch", "theater curtain",
    "thimble", "thresher", "throne", "tile roof", "toaster",
    "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck",
    "toyshop", "tractor", "trailer truck", "tray", "trench coat",
    "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus",
    "trombone", "tub", "turnstile", "typewriter keyboard", "umbrella",
    "unicycle", "upright", "vacuum", "vase", "vault",
    "velvet", "vending machine", "vestment", "viaduct", "violin",
    "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe",
    "warplane", "washbasin", "washer", "water bottle", "water jug",
    "water tower", "whiskey jug", "whistle", "wig", "window screen",
    "window shade", "Windsor tie", "wine bottle", "wing", "wok",
    "wooden spoon", "wool", "worm fence", "wreck", "yawl",
    "yurt", "web site", "comic book", "crossword puzzle", "street sign",
    "traffic light", "book jacket", "menu", "plate", "guacamole",
    "consomme", "hot pot", "trifle", "ice cream", "ice lolly",
    "French loaf", "bagel", "pretzel", "cheeseburger", "hotdog",
    "mashed potato", "head cabbage", "broccoli", "cauliflower", "zucchini",
    "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke",
    "bell pepper", "cardoon", "mushroom", "Granny Smith", "strawberry",
    "orange", "lemon", "fig", "pineapple", "banana",
    "jackfruit", "custard apple", "pomegranate", "hay", "carbonara",
    "chocolate sauce", "dough", "meat loaf", "pizza", "potpie",
    "burrito", "red wine", "espresso", "cup", "eggnog",
    "alp", "bubble", "cliff", "coral reef", "geyser",
    "lakeside", "promontory", "sandbar", "seashore", "valley",
    "volcano", "ballplayer", "groom", "scuba diver", "rapeseed",
    "daisy", "yellow lady's slipper", "corn", "acorn", "hip",
    "buckeye", "coral fungus", "agaric", "gyromitra", "stinkhorn",
    "earthstar", "hen-of-the-woods", "bolete", "ear", "toilet tissue",
};

std::string get_class_name(int id) {
    if (id >= 0 && id < (int)IMAGENET_CLASSES.size()) return IMAGENET_CLASSES[id];
    return "class_" + std::to_string(id);
}

// ============================================================================ 
// Utility Functions
// ============================================================================ 

cv::Mat preprocess_image(const cv::Mat& img, int input_h, int input_w, float& x_scale, float& y_scale) {
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat result;
    
    if (PREPROCESS_TYPE == LETTERBOX_TYPE) {
        float scale = std::min(1.0f * input_h / img.rows, 1.0f * input_w / img.cols);
        x_scale = scale; y_scale = scale;
        
        int new_w = static_cast<int>(img.cols * scale);
        int new_h = static_cast<int>(img.rows * scale);
        
        // Alignment usually handled by simple resize, but strict letterbox:
        int x_shift = (input_w - new_w) / 2;
        int y_shift = (input_h - new_h) / 2;
        
        cv::resize(img, result, cv::Size(new_w, new_h));
        cv::copyMakeBorder(result, result, y_shift, input_h - new_h - y_shift, 
                          x_shift, input_w - new_w - x_shift,
                          cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
    } else {
        cv::resize(img, result, cv::Size(input_w, input_h));
        x_scale = 1.0f * input_w / img.cols;
        y_scale = 1.0f * input_h / img.rows;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    LOG_TIME("Preprocess time", duration);
    return result;
}

cv::Mat bgr2nv12(const cv::Mat& bgr_img) {
    auto start = std::chrono::high_resolution_clock::now();
    
    int h = bgr_img.rows, w = bgr_img.cols;
    cv::Mat yuv;
    cv::cvtColor(bgr_img, yuv, cv::COLOR_BGR2YUV_I420);
    cv::Mat nv12(h * 3 / 2, w, CV_8UC1);
    
    uint8_t* y_ptr = nv12.ptr<uint8_t>();
    uint8_t* uv_ptr = y_ptr + h * w;
    uint8_t* u_src = yuv.ptr<uint8_t>() + h * w;
    uint8_t* v_src = u_src + (h/2) * (w/2);
    
    // Copy Y
    memcpy(y_ptr, yuv.ptr<uint8_t>(), h * w);
    
    // Interleave U and V for NV12
    for (int i = 0; i < (h/2) * (w/2); ++i) {
        *uv_ptr++ = *u_src++;
        *uv_ptr++ = *v_src++;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    LOG_TIME("BGR to NV12 time", duration);
    return nv12;
}

std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> result(logits.size());
    float max_val = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); i++) {
        result[i] = std::exp(logits[i] - max_val);
        sum += result[i];
    }
    for (size_t i = 0; i < logits.size(); i++) result[i] /= sum;
    return result;
}

// ============================================================================ 
// Main
// ============================================================================ 

int main(int argc, char** argv) {
    LOG_INFO("=== Ultralytics YOLO Classify Demo (S100 Nash-E UCP) ===");
    LOG_INFO("OpenCV Version: " << CV_VERSION);
    
    // 1. Parse Args
    std::string model_path = MODEL_PATH;
    std::string test_img_path = TEST_IMG_PATH;
    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) test_img_path = argv[2];

    // 2. Init Model
    LOG_INFO("Loading model: " << model_path);
    auto start_load = std::chrono::high_resolution_clock::now();

    hbDNNPackedHandle_t packed_dnn_handle;
    const char* model_fn = model_path.c_str();
    CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_dnn_handle, &model_fn, 1), "Init failed");
    
    const char** model_name_list;
    int model_count = 0;
    CHECK_SUCCESS(hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle), "Get model name failed");
    if (model_count > 1) {
        LOG_WARN("Model file contains " << model_count << " models, using the first one");
    }
    
    hbDNNHandle_t dnn_handle;
    CHECK_SUCCESS(hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]), "Get model handle failed");

    auto dur_load = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_load).count() / 1000.0;
    LOG_TIME("Load model time", dur_load);

    // 3. Input Properties
    hbDNNTensorProperties in_props;
    CHECK_SUCCESS(hbDNNGetInputTensorProperties(&in_props, dnn_handle, 0), "Get input props failed");
    
    int input_h = in_props.validShape.dimensionSize[1];
    int input_w = in_props.validShape.dimensionSize[2];
    if (input_h <= 3 && in_props.validShape.dimensionSize[2] > 3) {
         input_h = in_props.validShape.dimensionSize[2];
         input_w = in_props.validShape.dimensionSize[3];
    }
    LOG_INFO("Model Input Shape: " << input_w << "x" << input_h);
    
    // 4. Load and Preprocess Image
    LOG_INFO("Loading image: " << test_img_path);
    cv::Mat img = cv::imread(test_img_path);
    if (img.empty()) { 
        LOG_ERROR("Failed to load image: " << test_img_path); 
        return -1; 
    }
    
    float x_scale, y_scale;
    cv::Mat preprocessed_img = preprocess_image(img, input_h, input_w, x_scale, y_scale);
    cv::Mat nv12_img_full = bgr2nv12(preprocessed_img);

    // 5. Prepare Input Tensors
    int32_t input_count = 0;
    CHECK_SUCCESS(hbDNNGetInputCount(&input_count, dnn_handle), "Get input count failed");
    std::vector<hbDNNTensor> input_tensors(input_count);

    for (int i = 0; i < input_count; ++i) {
        CHECK_SUCCESS(hbDNNGetInputTensorProperties(&input_tensors[i].properties, dnn_handle, i), "Get props failed");
        int data_size = 0;
        
        if (input_count > 1) { // S100 Split Input Mode
            if (i == 0) { // Y Plane
                data_size = input_h * input_w;
                // Manually set props for split input if needed
                input_tensors[i].properties.validShape.dimensionSize[0] = 1;
                input_tensors[i].properties.validShape.dimensionSize[1] = input_h;
                input_tensors[i].properties.validShape.dimensionSize[2] = input_w;
                input_tensors[i].properties.validShape.dimensionSize[3] = 1;
                input_tensors[i].properties.stride[3] = 1;
                input_tensors[i].properties.stride[2] = 1;
                input_tensors[i].properties.stride[1] = input_w;
                input_tensors[i].properties.stride[0] = input_h * input_w;
            } else { // UV Plane
                int uv_h = input_h / 2;
                int uv_w = input_w / 2;
                data_size = uv_h * uv_w * 2;
                
                input_tensors[i].properties.validShape.dimensionSize[0] = 1;
                input_tensors[i].properties.validShape.dimensionSize[1] = uv_h;
                input_tensors[i].properties.validShape.dimensionSize[2] = uv_w;
                input_tensors[i].properties.validShape.dimensionSize[3] = 2;
                input_tensors[i].properties.stride[3] = 1;
                input_tensors[i].properties.stride[2] = 2;
                input_tensors[i].properties.stride[1] = uv_w * 2;
                input_tensors[i].properties.stride[0] = uv_h * uv_w * 2;
            }
        } else { // Standard Single Input
             data_size = input_h * input_w * 3 / 2;
        }
        
        CHECK_SUCCESS(hbUCPMallocCached(&input_tensors[i].sysMem, data_size, 0), "Malloc failed");
        
        if (input_count > 1) {
             if (i == 0) memcpy(input_tensors[i].sysMem.virAddr, nv12_img_full.data, data_size);
             else memcpy(input_tensors[i].sysMem.virAddr, nv12_img_full.data + input_h * input_w, data_size);
        } else {
             memcpy(input_tensors[i].sysMem.virAddr, nv12_img_full.data, data_size);
        }
        
        hbUCPMemFlush(&input_tensors[i].sysMem, HB_SYS_MEM_CACHE_CLEAN);
    }

    // 6. Prepare Outputs
    int output_count = 0;
    CHECK_SUCCESS(hbDNNGetOutputCount(&output_count, dnn_handle), "Get output count failed");
    std::vector<hbDNNTensor> outputs(output_count);
    for (int i = 0; i < output_count; i++) {
        CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&outputs[i].properties, dnn_handle, i), "Get out props failed");
        CHECK_SUCCESS(hbUCPMallocCached(&outputs[i].sysMem, outputs[i].properties.alignedByteSize, 0), "Malloc out failed");
    }

    // 7. Infer
    LOG_INFO("Running inference...");
    auto start_infer = std::chrono::high_resolution_clock::now();
    
    hbUCPTaskHandle_t task_handle = nullptr;
    CHECK_SUCCESS(hbDNNInferV2(&task_handle, outputs.data(), input_tensors.data(), dnn_handle), "Infer failed");
    
    hbUCPSchedParam ctrl_param;
    HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
    ctrl_param.backend = HB_UCP_BPU_CORE_ANY;
    
    CHECK_SUCCESS(hbUCPSubmitTask(task_handle, &ctrl_param), "Submit failed");
    CHECK_SUCCESS(hbUCPWaitTaskDone(task_handle, 0), "Wait failed");
    
    auto dur_infer = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_infer).count() / 1000.0;
    LOG_TIME("Infer time", dur_infer);

    // 8. Post Process
    LOG_INFO("Post-processing...");
    auto start_post = std::chrono::high_resolution_clock::now();
    
    CHECK_SUCCESS(hbUCPMemFlush(&outputs[0].sysMem, HB_SYS_MEM_CACHE_INVALIDATE), "Flush output failed");
    
    float* data = reinterpret_cast<float*>(outputs[0].sysMem.virAddr);
    int num_classes = outputs[0].properties.validShape.dimensionSize[1];
    
    // Handle case where dims might be [1, 1, 1, classes] or [1, classes, 1, 1]
    if (num_classes == 1) num_classes = outputs[0].properties.validShape.dimensionSize[3]; 
    if (num_classes == 1) num_classes = outputs[0].properties.validShape.dimensionSize[2];
    
    std::vector<float> logits(data, data + num_classes);
    std::vector<float> probs = softmax(logits);

    std::vector<int> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + TOP_K, indices.end(),
                      [&probs](int a, int b) { return probs[a] > probs[b]; });

    auto dur_post = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_post).count() / 1000.0;
    LOG_TIME("Post-processing time", dur_post);

    for (int i = 0; i < TOP_K; i++) {
        int idx = indices[i];
        std::cout << "TOP" << i+1 << ": " << get_class_name(idx) 
                  << " (id=" << idx << ", score=" << std::fixed << std::setprecision(3) 
                  << probs[idx] << ")" << std::endl;
    }

    // 9. Cleanup
    hbUCPReleaseTask(task_handle);
    for(auto& t : input_tensors) hbUCPFree(&t.sysMem);
    for(auto& out : outputs) hbUCPFree(&out.sysMem);
    hbDNNRelease(packed_dnn_handle);
    
    LOG_INFO("=== Demo completed successfully ===");
    return 0;
}
