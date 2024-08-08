/* initialize jsPsych */
var jsPsych = initJsPsych({
    on_finish: function() {
      jsPsych.data.displayData();
      //var full = jsPsych.data.get();
      //console.log(full);
    }
});
  
/* create timeline */
var timeline = [];

// Prolific ID input slide
var prolific_id_input = {
  type: jsPsychSurveyText,
  preamble: '<p>Before we begin, please enter your Prolific ID</p>',
  questions: [{prompt: 'Prolific ID:', name: 'prolific_id', required: true}],
  on_finish: function(data) {
    var prolific_id = data.response.prolific_id;
    jsPsych.data.addProperties({prolific_id: prolific_id});
  }
};

// Add the Prolific ID input slide to the timeline
timeline.push(prolific_id_input);
  
// debriefing form
function debriefing_form() {
    var content = "<div style='text-align:left; width:700px; margin:0 auto'>"
    + "<h3>Great work. Finally, we just have a couple of questions for you!</h3>"
    + "<p>Did you read and understand the instructions correctly?<br><input required='true' type='radio' id='yes_understood' name='understand' value='yes'><label for='yes'>Yes</label><br><input required='true' type='radio' id='not_understood' name='understand' value='no'><label for='no'>No</label><br></p>"
    + "<p>Were there any problems or bugs in the study?<br><input required='true' name='problems' type='text' size='50' style='width:100%;border-radius:4px;padding:10px 10px;margin:8px 0;border:1px solid #ccc;font-size:15px'/></p>"
    + "<p>Age:<br><input required='true' name='age' type='number' style='width:20%;border-radius:4px;padding:10px 10px;margin:8px 0;border:1px solid #ccc;font-size:15px'/></p>"
    + "<p>Please indicate your gender:<br><input required='true' type='radio' id='male' name='gender' value='male'><label for='male'>Male</label><br><input required='true' type='radio' id='female' name='gender' value='female'><label for='female'>Female</label><br><input required='true' type='radio' id='other' name='gender' value='other'><label for='other'>Other</label></p>"
    + "<p>Any additional comments you would like to share?<br><input name='comments' type='text' size='50' style='width:100%;border-radius:4px;padding:10px 10px;margin:8px 0;border:1px solid #ccc;font-size:15px'/></p>";
    return content;
};
  
// generate a random subject ID with 8 characters
var subject_id = jsPsych.randomization.randomID(9);
  
jsPsych.data.addProperties({ID: subject_id});
  
var images = [
    'img/map1.png', 'img/map2.png', 'img/map3.png',
    ];
    
var sign_with_rule = 'img/sign.png'

var all_images = images.concat(sign_with_rule)
 
var preloadAll = {
    type: jsPsychPreload,
    images: all_images,
    auto_preload: true 
};
timeline.push(preloadAll);
  
/* define welcome message trial */
var welcome = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: '<p style="text-align:center;width:700px;"> Welcome to the experiment! This study takes most people around 5 minutes to complete. Please complete it in one sitting.</p>',
    choices: [" "],
    prompt: 'Press \'Spacebar\' to continue.'
};
timeline.push(welcome);
  
/* define consent form */
var consent = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: '<p style = "font-size:16px;">MIT Computational Cognitive Science Lab</p><h4>Press the \'y\' key to consent.</h4><div style = "background-color:#f2f2f2;"><h3>Informed Consent Form</h3><div style="text-align:left;width:700px;"><p style = "font-size:16px;">In this study, you will be asked to read a set of instructions that describe a particular scenario, and you will then be asked to perform a simple task relating to that scenario. There will also be comprehension and attention checks. Participants who do not answer these checks correctly cannot be approved. In order to get credit for this study, you must read all the questions fully and answer honestly. You may only complete this study once. If you complete this study multiple times, you will be rejected.</p><p style = "font-size:16px;">This study is part of a MIT scientific research project. Your decision to complete this study is voluntary. There is no way for us to identify you. The only information we will have, in addition to your responses, is the time at which you completed the survey. The results of the research may be presented at scientific meetings or published in scientific journals. Pressing the "Y" key indicates that you are at least 18 years of age, and agree to complete this study voluntarily. Press the "Y" key to confirm your agreement and continue.</p></div>',
    choices: ['y'],
};
timeline.push(consent);
  
/* define instructions trial */
var instructions1 = {
    type: jsPsychSurveyText,
    preamble: `
    <p>(Instructions 1 out of 2)</p>
    <p style="text-align:center;width:900px;"> <h4>Welcome! We are conducting an experiment to gather information about how people reason about rules that are posted in the world.</h4> </p>
    <p style="text-align:center;width:900px;"> <h4>In this experiment, you will be looking at maps which depict a park and the surrounding areas in a medium-sized town. Every day, about 100 people go through this area on their way to various destinations.</h4> </p>
    <p style="text-align:center;width:900px;"> <h4>This map depicts three common <i>destinations</i> that people in this town might be going to when they are traveling through this area, as well as the different kinds of <i>terrain</i> that people need to travel through when traveling in this area.</h4> </p>

    `,  
    questions:[{prompt: 'What does the map that you will see represent?', required: true, name: 'instructions1'}],
    data:{trial_category: "instructions"}
};
timeline.push(instructions1); 
  
var instructions2 = {
    type: jsPsychSurveyText,
    preamble: `
    <p>(Instructions 2 out of 2)</p>
    <p style="text-align:center;width:900px;"> <h4>Here is a legend showing the three possible destinations, and the different kinds of terrain.</h4> </p>
    <p><img style="height:auto;width:600px;" src='img/Key.png'></img></p>
    <p style="text-align:center;width:900px;">In this experiment, you will see 3 different maps such as the one below. For each map, we will ask you to answer a series of questions about the map, a sign in the park, and the people in the town. There are no right answers. Please look at each map, then answer each question with your own best judgment based on the map that you see, as well as your own common sense intuitions about towns and these kinds of destinations in general.</p>
    <p><img style="height:auto;width:600px;" src='img/map1.png'></img></p>
    <p style="text-align:center;width:900px;">When you are ready, please type <strong>\"Yes\"</strong> below to complete a quick comprehension check, before moving on to the experiment.</p>
    `,
    questions:[{prompt: 'Are you ready?', required: true, name: 'instructions2'}],
    data:{trial_category: "instructions"}
};
timeline.push(instructions2);

var instructions3 = {
  type: jsPsychSurveyMultiChoice,
  questions: [
    {
      prompt: "What will you be looking at in this experiment??", 
      name: 'map_attention', 
      options: ['A person traveling through the park', 'Maps showing a small city, park, and destinations', 'A group of people traveling through a small city'], 
      required: true,
      horizontal: true
    }, 
    {
      prompt: "What do the different squares on the map represent?", 
      name: 'squares_attention', 
      options: ['Two terrains (grass and sidewalk) and three destinations (urgent care, bus stop, coffee shop)', 'Two terrains (lava and floor) and three destinations (urgent care, bus stop, coffee shop)', 'Two terrains (grass and sidewalk) and three destinations (school, train station, ice cream shop)'], 
      required: true,
      horizontal: true
    },
    {
      prompt: "How should you answer each question in this experiment?", 
      name: 'answer_attention', 
      options: ['Based on what you think other participants are most likely to agree with.', 'Based on your own best judgment based on the map that you see, as well as your <i>own</i> common sense intuitions about towns and these kinds of destinations in general.', 'Based on what you think the average person living in this town would think'], 
      required: true,
      horizontal: true
    }
  ],
};
timeline.push(instructions3);

// Function to create HTML stimulus with a statement and two images
function createStimulusHtml(mainImage) {
    var html = "<div style='text-align:center;'>"
        + "<p>In this park, there is a sign which reads 'Do not walk on the grass'.</p>"
        + "<div style='display:flex; justify-content:center; align-items:center;'>"
        + "<img src='" + sign_with_rule + "' style='width:140px; height:auto; margin-right:10px;' />" // Smaller sign image
        + "<img src='" + mainImage + "' style='width:460px; height:auto;' />" // Larger main image
        + "</div>"
        + "</div>";
    return html;
}

// Define new slider scale questions
var sliderQuestions = [
    {
        prompt: "How important is it to go to the bus stop?",
        labels: ['Not important at all', 'Moderately important', 'Extremely important']
    },
    {
        prompt: "How important is it to go to the urgent care center?",
        labels: ['Not important at all', 'Moderately important', 'Extremely important']
    },
    {
        prompt: "How important is it to go to the coffee shop?",
        labels: ['Not important at all', 'Moderately important', 'Extremely important']
    }
];

// Function to create slider scale trial
function createSliderTrial(image, question) {
    return {
        type: jsPsychHtmlSliderResponse,
        stimulus: createStimulusHtml(image),
        labels: question.labels,
        prompt: '<p>' + question.prompt + '</p>',
        slider_start: 50,
        min: 0,
        max: 100,
        step: 1,
        require_movement: true,
        data: { trial_category: 'slider_trial', stimulus_name: image, question: question.prompt }
    };
}

// Generate slider scale trials for each image
var sliderTrials = [];
images.forEach(function(image) {
    sliderQuestions.forEach(function(question) {
        sliderTrials.push(createSliderTrial(image, question));
    });
});

// Existing free-response questions
var questions = [
    "Why do you think they have this rule?",
    "How many people do you think walk in this park on a single day?",
    "How many people do you think will go to the bus stop today?",
    "How many people do you think will go to the coffee shop today?",
    "How many people do you think will go to the urgent care center today?",
    "How bad would it be for everyone if the grass died?",
    "How much do you think the sign-writer wanted the grass to stay alive?",
    "What would you guess is the number of people stepping on the grass for it to die?",
    "Who do you think is most likely wrote the sign with the rule? A council member who polled the community and measured and analyzed the traffic at the park? A person randomly selected from the community? A person from the community who doesn't really visit the park? Or someone else?"
];

// Create free-response trials
function createImageQuestionTrial(image, question) {
    return {
        type: jsPsychSurveyText,
        preamble: createStimulusHtml(image),
        questions: [{ prompt: question, rows: 5, columns: 40, required: true }],
        data: {
            trial_category: 'image_question_trial',
            stimulus_name: image,
            question_text: question
        }
    };
}

// Generate free-response trials
var imageTrials = [];
images.forEach(function(image) {
    questions.forEach(function(question) {
        imageTrials.push(createImageQuestionTrial(image, question));
    });
});

// Combine free-response and slider scale questions
var allQuestions = imageTrials.concat(sliderTrials);

// Shuffle all questions
var shuffledAllQuestions = jsPsych.randomization.shuffle(allQuestions);

// Add shuffled questions to the timeline
shuffledAllQuestions.forEach(function(trial) {
    timeline.push(trial);
});

/* define debrief */
var debrief_qs = {
    type: jsPsychSurveyHtmlForm,
    html: debriefing_form(),
    data: {trial_category: 'debrief'},
    dataAsArray: true,
    on_finish: function(data){}
};
timeline.push(debrief_qs);
  
var conclusion = {
    type: jsPsychSurveyText,
    preamble: '<div style="font-size:20px;">This task is now over. The completion code is: ' + subject_id + ' Thank you for your participation! </div>',
    questions: [{prompt: 'confirm by entering your code here, and press continue', required: true}]
};
timeline.push(conclusion);
  
/* start the experiment */
jsPsych.run(timeline);