#!/bin/bash

#This script is used to calibrate pulsar data.
#First we need to make sure that a calibrator (noise diode) file exists. Should be either a file with _R or ending in .sf.
#The calibrator needs to be prepared:
#for example:

#Fold the calibrator data. The noise diode is fired at 1Hz rate. Get 4s subints.
#dspsr -scloffs -c 1.0 -t 8 -L 4 -A 2024-03-01-07:42:10.sf

#Clean the calibrator data and get the channel mask too
#paz -z "432 415 416 425" -w "8 5 29 18" -e pazi 2024-03-01-07:42:08.ar

#Check if the calibrator looks good. But before that make sure that the header is set correctly.
#psredit -m -c "type=PolnCal" -c "rcvr:basis=lin" -c "rcvr:name=P170" -c "name=B0355+54_R" -c "be:name=EDD" 2024-03-01-07:42:08.pazi

#Convert to PSRFITS
#psrconv -o PSRFITS 2024-03-01-07:42:08.pazi

#Now check the polarisation cal solutions and remove any channels are outliers ...
#pacv -d 2024-03-01-07:42:08.cf

#Now we can work on the data:
# Get the current directory where the script is located
directory="$(pwd)"

#We first apply the same mask that was applied to the calibrator to all the data (NEEDS TO BE UPDATED IF USING A NEW CALIBRATOR!!!). Make sure not to include the weights if calibrating single pulses or specific subints.
paz -z "158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 156 157 201 202 203 204 205 234 235 236 237 238 239 240 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 38 45 46 47 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 383 382 390 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 453 454 455 456 457 458 459 460 461 462 463 464 251 253 254 255 80 81 82 87 88 89 90 91 92 93 94 95 96 97 98 49 141 142 143 144 145 146 332 395 397 398 440 441 377 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402" -e autopazi "$directory"/*.clean

#We then add all subints (if there are any, step is optional, it is more to check)
#pam -T -e autoTT *.autopazi

#Edit headers, and make sure everything is fine
psredit -m -c "type=Pulsar" -c "rcvr:basis=lin" -c "rcvr:name=P170" -c "be:name=EDD" *.autopazi

#Convert data to PSRFITS
psrconv -o PSRFITS *.autopazi

#Create the polarisation calibration database using the calibrator
pac -P -w *.cf

#Apply polcal on data now - database.txt file was created by the above step
pac -P -d database.txt *.rf

#Apply the RM correction
pam --RM 79 -e rm *.calibP

#Remove extra files
rm *.autopazi
rm *.rf
#rm *.calibP
#rm *.autoTT

#Create added and scrunched files
#Add files into one time scrunched file
psradd -T -o added_calibrated.TT *.rm

#Add files into one frequency scrunched by first individually scrunching each single pulse in frequency and then adding them up.
pam -F -e .autoFF *.rm
psradd -o added_calibrated.FF *.autoFF
#Remove extra files
rm *.autoFF
